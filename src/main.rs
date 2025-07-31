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

use clap::Parser;
use reqwest;
use std::env;
use std::fs;
use std::io;
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

fn get_system_message(system_message: &str) -> String {
    // Condition 1: Check custom prompt file in XDG config directory
    let home = env::var("HOME")
        .expect("Could not retrieve system_message: HOME enviroment variable not set.");
    if !system_message.contains('/') {
        let mut path_buf = PathBuf::from(home);
        path_buf.push(".config");
        path_buf.push("rapidllm");
        path_buf.push("prompts");
        path_buf.push(system_message);
        path_buf.push("system.md");

        if let Ok(content) = fs::read_to_string(&path_buf) {
            return content;
        }
    }

    // Condition 2: Check if input is a valid file path
    if let Ok(content) = fs::read_to_string(system_message) {
        return content;
    }

    // Condition 3: Return original string
    system_message.to_string()
}

fn get_api_key() -> std::string::String {
    let home =
        env::var("HOME").expect("Could not retrieve API key: HOME enviroment variable not set.");
    let path = Path::new(&home)
        .join(".config")
        .join("rapidllm")
        .join("openrouter")
        .join("api_key");

    // more verbose messages (e.g. "No such file or directory.")
    read_to_string(path).unwrap_or_else(|err| {
        eprintln!(
            "Could not read OpenRouter API key at ~/.config/rapidllm/openrouter/api_key: {}",
            err
        );
        std::process::exit(1);
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.license {
        println!("GNU LGPLv3+");
        return Ok(());
    }

    if args.verbose {
        eprintln!("rlm started");
    }

    let stdin = io::stdin();

    let api_key = get_api_key();

    if args.verbose {
        eprintln!("Read OpenRouter API key.");
    }

    // retrieve user message
    let input = io::read_to_string(stdin).expect("Failed to read input from stdin");

    let user_message = input.trim().to_string();
    if args.verbose {
        eprintln!("Read input of size {}", user_message.len());
    }

    // retrieve system message
    let is_system_message_present = args.system != None;
    let mut system_message: String = String::new();
    if is_system_message_present {
        system_message = get_system_message(args.system.unwrap().trim());
        if args.verbose {
            eprintln!(
                "Read system message:\n\n```\n{}\n```\n\n...of size {}",
                system_message,
                system_message.len()
            );
        }
    }

    // handle input size errors
    if user_message.len() + system_message.len() == 0 {
        println!("Input and system message are empty");
        std::process::exit(1);
    }

    if user_message.len() + system_message.len() > args.character_limit {
        eprintln!(
            "Input too long: {} characters given, but the limit is {}",
            user_message.len() + system_message.len(),
            args.character_limit
        );

        std::process::exit(1);
    }

    let mut request_body = OpenRouterRequest {
        model: String::from(args.model),
        messages: Vec::<Message>::new(),
    };

    if is_system_message_present {
        if args.raw_request {
            eprintln!("System message: {}", system_message);
        }
        request_body.messages.push(Message {
            role: "system".to_string(),
            content: system_message,
        });
    }

    if args.raw_request {
        eprintln!("User message: {}", user_message);
    }
    request_body.messages.push(Message {
        role: "user".to_string(),
        content: user_message,
    });

    let client = reqwest::Client::new();
    let response = client
        .post("https://openrouter.ai/api/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request_body)
        .send()
        .await?;

    // Check if the response status is successful
    if !response.status().is_success() {
        let status = response.status();
        let response_text = response.text().await?;
        eprintln!("API request failed with status: {}", status);
        eprintln!("Response body: {}", response_text);
        std::process::exit(1);
    }

    let response_text = response.text().await?;
    let response_json: OpenRouterResponse = match serde_json::from_str(&response_text) {
        Ok(json) => json,
        Err(e) => {
            eprintln!("Failed to parse JSON response: {}", e);
            eprintln!("Response body was: {}", response_text);
            std::process::exit(1);
        }
    };

    if let Some(first_choice) = response_json.choices.first() {
        print!("{}", first_choice.message.content);
    } else {
        eprintln!("No response from LLM API");
        std::process::exit(1);
    }

    Ok(())
}
