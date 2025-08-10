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
use std::{fs::read_to_string, path::Path};

#[derive(serde::Serialize)]
struct OpenRouterRequest {
    model: String, // e.g., "mistralai/mistral-7b-instruct"
    messages: Vec<Message>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Message {
    role: String, // "user", "system", etc.
    content: String,
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

    let mut request_body = OpenRouterRequest {
        model: String::from(args.model),
        messages: Vec::<Message>::new(),
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
        content: user_message,
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
            content: system_message,
        });
    }

    if args.raw_request {
        let mut size = 0;
        for message in &request_body.messages {
            eprintln!("{}:{}", message.role, message.content);
            size += message.content.len();
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
    }

    let client = reqwest::Client::new();
    let response = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request_body)
        .send()
        .await
        .context("Failed to send API request")?;

    // Check if the response status is successful
    if !response.status().is_success() {
        let status = response.status();
        let response_text = response.text().await?;
        return Err(anyhow::anyhow!(format!(
            "API responeded with status {}; Response body was: {}",
            status, response_text
        )));
    }

    let response_text = response.text().await?;
    let response_json: OpenRouterResponse = match serde_json::from_str(&response_text) {
        Ok(json) => json,
        Err(e) => {
            return Err(e).context(format!(
                "Failed to parse JSON of the API request response; Response body was: {}",
                response_text
            ));
        }
    };

    let first_choice = response_json
        .choices
        .first()
        .context("No response from LLM API")?;

    print!("{}", first_choice.message.content);
    Ok(())
}
