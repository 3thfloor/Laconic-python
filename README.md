# thirthfloor

Local AI inference for Python. One install. No daemon. No API keys.

Your model runs as an object inside your process. Ask a question, get a string back.

```python
from thirthfloor import Engine

engine = Engine()
engine.load("qwen", "/models/qwen3-4b-q4.gguf")
print(engine.chat("qwen", "What is boundary value analysis?"))
```

## Install

```bash
pip install 3thfloor
```

CPU works out of the box. For GPU acceleration, install the matching wheel for your hardware:

**Apple Silicon (Metal)**

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/metal
pip install 3thfloor
```

**NVIDIA (CUDA 12.x)**

```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
pip install 3thfloor
```

Same package, same API. The backend picks up your GPU automatically.

## Quick Start

```python
from thirthfloor import Engine

engine = Engine()
engine.load("tester", "/models/3thfloor-tester-q4.gguf")

answer = engine.chat("tester", "Write three test cases for a login form.")
print(answer)
```

`chat()` returns a plain string. Not a dict, not `.choices[0].message.content`. A string.

## Sessions (Conversation History)

```python
session = engine.session("tester", system="You are a senior QA engineer.")

print(session.send("What is exploratory testing?"))
print(session.send("How is that different from what I asked about?"))
```

The session tracks history for you. The second question knows about the first. No message arrays to build, no history to splice.

## Streaming

```python
for token in engine.stream("tester", "Explain risk-based testing in two paragraphs."):
    print(token, end="", flush=True)
```

`stream()` yields token strings as they generate. Print them, pipe them, collect them.

## Multiple Models

```python
engine.load("fast", "/models/qwen3-4b-q4.gguf")
engine.load("smart", "/models/qwen3-32b-q4.gguf")

def ask(question: str) -> str:
    alias = "smart" if len(question) > 200 else "fast"
    return engine.chat(alias, question)
```

Load as many models as your RAM allows. Route between them with the alias.

## Agents

```python
from thirthfloor import tool, run_agent

@tool
def get_build_status(pipeline: str) -> str:
    """Return the latest CI status for a pipeline."""
    return "pipeline main: passing, 312 tests green"

result = run_agent(
    engine, "tester",
    "Is the main pipeline healthy?",
    tools=[get_build_status],
)
print(result)
```

Decorate a function with `@tool`, pass it to `run_agent()`. The model decides when to call it, the engine runs it, you get the final answer as a string. Docstrings and type hints become the tool schema.

## Model Management

```python
engine.models.add("tester", "/models/3thfloor-tester-q4.gguf")

for m in engine.models.list():
    print(m["alias"], m["path"], round(m["size_mb"] / 1024, 1), "GB")

engine.models.download("Qwen/Qwen3-4B-GGUF", filename="qwen3-4b-q4_k_m.gguf")
```

Registered models let you look up the path by alias: `engine.load("tester", engine.models.info("tester")["path"])`.

Downloading from HuggingFace requires the extra:

```bash
pip install "thirthfloor[hf]"
```

## Optional HTTP Server

```python
engine.serve(port=7437)
```

OpenAI-compatible endpoints (`/v1/chat/completions`) for when other tools need HTTP access. Never required. Requires:

```bash
pip install "thirthfloor[server]"
```

## Embed in Your Software

```python
from thirthfloor import Engine

class SupportBot:
    def __init__(self, model_path: str):
        self.engine = Engine()
        self.engine.load("support", model_path)
        self.session = self.engine.session(
            "support",
            system="You answer questions about our test automation product.",
        )

    def reply(self, message: str) -> str:
        return self.session.send(message)

bot = SupportBot("/models/3thfloor-tester-q4.gguf")
print(bot.reply("How do I tag a flaky test?"))
```

The model is an object in your process. No ports to manage, no subprocesses to babysit, no service that has to be running before your app starts. When your process exits, the model is gone. Ship it inside a CLI, a desktop app, a batch job, anywhere Python runs.

---

Built by Justin Bench, [3th Floor AI](https://3thfloor.com).

Free for personal projects, research, experiments, and noncommercial use under the [PolyForm Noncommercial License 1.0.0](./LICENSE). Commercial use requires a license: justin@3thfloor.com.
