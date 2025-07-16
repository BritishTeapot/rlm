# rlm

A simple shell command that recieves text in `stdin` and outputs the response in `stdout`. Supports system messages and prompts. Now only supports OpenRouter, due to it's popularity and versatility.

`rlm` is build to be versatile and simple. It only does the bare minimum of what is required to interface with LLMs. You can use it to build more complicated scripts and systems with ease.

See [RapidLLM Manifesto](https://github.com/BritishTeapot/rapidllm.git)

# Usage

Use it like any other standard core shell commands. Think of it as `grep`: pipe in the question, get the answer piped out.

For example, the command:

```bash
echo "Why is open-source superior to proprietary software?" | rlm
```

Will send a request to OpenRouter, with just one user message in the context:

```
User: Why is open-source superior to proprietary software?
```

And will output whatever the model's response was **without thinking**.

You can change the used model with `--model` flag. For example...

```bash
echo "Why is open-source superior to proprietary software?" | rlm --model "deepseek/deepseek-r1-0528"
```

...will send a request to OpenRouter for the DeepSeek R1 0528 model.

## Prompts

`rlm` has a `--prompt` flag, that works as follows:

1. if text after `--prompt` corresponds to a name of existent directory in `~/.config/rapidllm/prompts/`, and it contains contains `system.md` file, the model will receive the contents of that `system.md` file as a system message, prior to the user message.
2. otherwise, if text after `--prompt` corresponds to a name of a file, the model will receive the contents of that file as a system message, prior to the user message.
3. otherwise, the model will receive the text after `--prompt` as a system message, prior to the user message.

For example...

```bash
echo "Why is open-source superior to proprietary software?" | rlm --prompt ~/Documents/powerful_prompt.txt
```

...will send the following request to OpenRouter:

```
System: [contents of ~/Documents/powerful_prompt.txt]
User: Why is open-source superior to proprietary software?
```

And...

```bash
echo "Why is open-source superior to proprietary software?" | rlm --prompt "You are a helpful assistant."
```

... will send:

```
System: You are a helpful assistant.
User: Why is open-source superior to proprietary software?
```

# Building and Running

Use cargo.

```
cargo build
```

```
cargo run
```

# Installation

Use cargo.

```
cargo install --path .
```

# License

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