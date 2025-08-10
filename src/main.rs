/*
rlm - core LLM command line interface of rapidllm.
Copyright (C) 2025 Fedir Kovalov

This program is free software: you can redistribute it
and/or modify it under the terms of the GNU Lesser
General Public License as published by the Free
Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will
be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License
for more details.

You should have received a copy of the GNU Lesser
General Public License along with this program. If not,
see <https://www.gnu.org/licenses/>.
*/

use anyhow::{Context, Result};
use clap::Parser;
use reqwest;
use std::env;
use std::fs;
use std::io;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::process::Command;
use std::{fs::read_to_string, path::Path};

#[derive(serde::Serialize)]
struct OpenRouterRequest {
    model: String, // e.g., "mistralai/mistral-7b-instruct"
    messages: Vec<Message>,
    tools: Option<Vec<ToolDefinition>>,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct FunctionObject {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct ToolDefinition {
    r#type: String,
    function: FunctionObject,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct ToolCall {
    id: String,
    r#type: String, // should be "function"
    function: ToolFunctionCall,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct ToolFunctionCall {
    name: String,
    arguments: String, // JSON string
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct Message {
    role: String,
    content: Option<String>, // can be null for assistant with tool_calls
    tool_calls: Option<Vec<ToolCall>>,
    tool_call_id: Option<String>,
}

#[derive(serde::Deserialize)]
struct OpenRouterResponse {
    choices: Vec<Choice>,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Parser)]
#[command(about = "rapidllm core command.", version)]
struct Args {
    /// AI model to use
    #[arg(short = 'm', long = "model", default_value = "thudm/glm-4-32b:free")]
    model: String,

    #[arg(short = 'c', long = "character_limit", default_value = "16384")]
    character_limit: usize,

    /// System prompt (optional)
    #[arg(short, long)]
    system: Option<String>,

    /// Directory path of the tool to make available for function calling
    #[arg(short = 't', long = "tool")]
    tool_dir: Option<PathBuf>,

    #[arg(long)]
    license: bool,

    #[arg(long)]
    raw_request: bool,

    #[arg(long)]
    verbose: bool,
}

fn get_system_message(system_message: &str) -> Result<String> {
    // Condition 1: Check custom prompt file in XDG config directory
    if !system_message.contains('/') {
        // if the string does not contain '/', then it cannot escape outside the promopts directory
        // (e.g. you cannot read ../../etc/passwd through system_message). This should be enough
        // since:
        // 1. '/' is one of the only characters no filaname on Linux can use.
        // 2. We are not targeting windows systems :) No need to worry about backslashes

        // If HOME isn't set, we should fail. We don't want situations where we interpret --system
        // parameter in a way that user did not intend.
        let home = env::var("HOME").context("HOME enviroment variable not set.")?;

        let mut path_buf = PathBuf::from(home);
        path_buf.push(".config");
        path_buf.push("rapidllm");
        path_buf.push("prompts");
        path_buf.push(system_message);
        path_buf.push("system.md");

        match std::fs::read_to_string(&path_buf) {
            Ok(content) => return Ok(content),

            Err(e) => {
                if e.kind() != ErrorKind::NotFound {
                    // if the function failed for any other reason than ENOENT, we should inform
                    // the user by erroring out
                    return Err(e).context(format!(
                        "Could not open file {}",
                        path_buf.display().to_string()
                    ));
                }
            }
        }

        // if the path was invalid (and no other error occured), the system_message might still refer to either a file in pwd or the
        // system message itself
    }

    // if the system_message does have '/', then it is either:
    // 1. A filepath.
    // 2. A system message that itself contains '/'.

    // Condition 2: Check if input is a valid file path
    match fs::read_to_string(system_message) {
        Ok(content) => return Ok(content),

        Err(e) => {
            if e.kind() != ErrorKind::NotFound {
                return Err(e).context("Could not open file");
            }
        }
    }

    // Condition 3: Return original string
    Ok(system_message.to_string())
}

fn get_api_key() -> Result<std::string::String> {
    let home = env::var("HOME").context("HOME enviroment variable not set.")?;

    let path = Path::new(&home)
        .join(".config")
        .join("rapidllm")
        .join("openrouter")
        .join("api_key");

    // more verbose messages (e.g. "No such file or directory.")
    read_to_string(path).context("Could not read ~/.config/rapidllm/openrouter/api_key")
}

fn get_user_message() -> Result<String> {
    let stdin = io::stdin();
    // retrieve user message, explicit failure if input is non-UTF8
    let input = match io::read_to_string(stdin) {
        Ok(read) => read,
        Err(e) => return Err(e).context("Could not read from stdin"),
    };

    Ok(input.trim().to_string())
}

fn load_tool_definition(tool_dir: &Path) -> Result<ToolDefinition> {
    let definition_path = tool_dir.join("definition.json");
    let content = fs::read_to_string(&definition_path).context(format!(
        "Failed to read tool definition at {:?}",
        definition_path
    ))?;
    let definition: ToolDefinition = serde_json::from_str(&content).context(format!(
        "Failed to parse JSON tool definition from {:?}",
        definition_path
    ))?;
    Ok(definition)
}

fn execute_tool(tool_dir: &Path, arguments: &str) -> Result<String> {
    let exec_path = tool_dir.join("exec");

    let output = Command::new(&exec_path)
        .arg(arguments)
        .output()
        .with_context(|| format!("Failed to execute tool at {:?}", exec_path))?;

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "Tool failed with exit status: {}",
            output.status
        ));
    }

    let stdout_str = String::from_utf8(output.stdout)
        .map_err(|_| anyhow::anyhow!("Tool output is not valid UTF-8"))?;

    Ok(stdout_str)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    if args.license {
        println!("GNU LGPLv3+");
        return Ok(());
    }

    if args.verbose {
        eprintln!("rlm started");
    }

    let api_key = get_api_key().context("Could not retrieve OpenRouter API key")?;
    if args.verbose {
        eprintln!("Read OpenRouter API key.");
    }

    let tool_definition = if let Some(tool_dir) = &args.tool_dir {
        Some(vec![load_tool_definition(tool_dir)?])
    } else {
        None
    };

    let mut request_body = OpenRouterRequest {
        model: args.model.clone(),
        messages: Vec::<Message>::new(),
        tools: tool_definition,
    };

    let user_message = get_user_message().context("Could not get user message")?;

    if args.verbose {
        eprintln!(
            "Read user message:\n\n```\n{}\n```\n\n...of size {}",
            &user_message,
            user_message.len()
        );
    }
    request_body.messages.push(Message {
        role: "user".to_string(),
        content: Some(user_message),
        tool_calls: None,
        tool_call_id: None,
    });

    // retrieve system message
    if let Some(system_message_arg) = args.system {
        let system_message = get_system_message(system_message_arg.trim())
            .context("Could not get system message")?;
        if args.verbose {
            eprintln!(
                "Read system message:\n\n```\n{}\n```\n\n...of size {}",
                &system_message,
                system_message.len()
            );
        }

        // push message into the message list
        request_body.messages.push(Message {
            role: "system".to_string(),
            content: Some(system_message),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    let mut size = 0;
    for message in &request_body.messages {
        if args.raw_request {
            if let Some(message_text) = message.content.clone() {
                size += message_text.len();
            }
        }
    }
    if size == 0 {
        return Err(anyhow::anyhow!("Input is empty"));
    }
    if size > args.character_limit {
        return Err(anyhow::anyhow!(format!(
            "Input too long: {} characters given, but the limit is {}",
            size, args.character_limit
        )));
    }

    if args.raw_request {
        let json_string = serde_json::to_string_pretty(&request_body)
            .context("Failed to serialize request to JSON")?;

        eprintln!("{}", json_string);
    }

    loop {
        let client = reqwest::Client::new();
        let response = client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Content-Type", "application/json")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&request_body)
            .send()
            .await
            .context("Failed to send API request")?;

        if !response.status().is_success() {
            let status = response.status();
            let response_text = response.text().await?;
            return Err(anyhow::anyhow!(format!(
                "API responded with status {}; Response body was: {}",
                status, response_text
            )));
        }

        let response_text = response.text().await?;
        let response_json: OpenRouterResponse = serde_json::from_str(&response_text)
            .with_context(|| format!("Failed to parse JSON response body: {}", response_text))?;

        let first_choice = response_json
            .choices
            .first()
            .context("No response from LLM API")?;

        // Step A: Append model's response to conversation history
        request_body.messages.push(first_choice.message.clone());

        // Step B: Check for function call
        if let Some(tool_calls) = &first_choice.message.tool_calls {
            let tool_dir = args.tool_dir.as_ref().unwrap(); // safe because we already handled this earlier

            for tool_call in tool_calls {
                if args.verbose {
                    eprintln!("Some tool called.");
                }
                if tool_call.r#type != "function" {
                    eprintln!("Unknown tool_call type: {}", tool_call.r#type);
                    continue;
                }

                let tool_def = &request_body.tools.as_ref().unwrap()[0]; // assuming single tool for now

                if args.verbose {
                    eprintln!("Tool {} called.", tool_call.function.name);
                }

                if tool_call.function.name != tool_def.function.name {
                    return Err(anyhow::anyhow!(
                        "Model called an unknown function: {}",
                        tool_call.function.name
                    ));
                }

                // Execute the function
                let tool_result_json = execute_tool(tool_dir, &tool_call.function.arguments)?;

                // Check character limit
                let mut total_chars = 0;
                for msg in &request_body.messages {
                    if let Some(content) = &msg.content {
                        total_chars += content.len();
                    }
                }

                if total_chars + tool_result_json.len() > args.character_limit {
                    return Err(anyhow::anyhow!(
                        "Function result too long: adding {} exceeds limit of {} characters",
                        tool_result_json.len(),
                        args.character_limit
                    ));
                }

                if args.raw_request {
                    let json_string = serde_json::to_string_pretty(&tool_result_json)
                        .context("Failed to serialize tool response to JSON")?;

                    eprintln!("{}", json_string);
                }

                // Push tool call result as tool role
                request_body.messages.push(Message {
                    role: "tool".to_string(),
                    content: Some(tool_result_json),
                    tool_calls: None,
                    tool_call_id: Some(tool_call.id.clone()),
                });

                // Continue looping to get next model response incorporating function result
            }
        } else {
            // No tool call -> print final response and break
            if let Some(content) = &first_choice.message.content {
                print!("{}", content);
            } else {
                return Err(anyhow::anyhow!(
                    "Model returned empty content after user input"
                ));
            }

            // Final check for overall input size too
            let mut total_size = 0;
            for msg in &request_body.messages {
                if let Some(content) = &msg.content {
                    total_size += content.len();
                }
            }
            if total_size == 0 {
                return Err(anyhow::anyhow!("Input is empty"));
            }
            if total_size > args.character_limit {
                return Err(anyhow::anyhow!(format!(
                    "Total message content size too large: {} > limit of {} characters",
                    total_size, args.character_limit
                )));
            }

            break;
        }
    }
    Ok(())
}
