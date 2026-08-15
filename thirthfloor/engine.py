"""3th Floor AI SDK engine.

The Engine is the main entry point. Load a GGUF model, chat with it,
stream tokens, or open a session with automatic history. No daemons,
no ports, no subprocesses. The model lives in memory as an object.

Author: Justin Bench, 3th Floor AI.
"""

import logging
import pathlib
import time

from llama_cpp import Llama

from .models import ModelManager
from .session import Session

logger = logging.getLogger("thirthfloor")


class Engine:
    """Loads GGUF models and talks to them.

    Usage:
        engine = Engine()
        engine.load("assistant", "/path/to/model.gguf")
        answer = engine.chat("assistant", "What is a closure?")
    """

    def __init__(self, models_dir="~/.3thfloor/models"):
        self.models_dir = pathlib.Path(models_dir).expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._loaded: dict = {}
        self._meta: dict = {}
        self.models = ModelManager(self)

    def load(self, alias, path, ctx=4096, gpu_layers=-1):
        """Load a GGUF model under an alias. Returns self for chaining."""
        if alias in self._loaded:
            raise ValueError(
                f"Alias '{alias}' is already loaded. "
                f"Call unload('{alias}') first or pick another alias."
            )
        model_path = pathlib.Path(path).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info("Loading model '%s' from %s", alias, model_path)
        model = Llama(
            model_path=str(model_path),
            n_ctx=ctx,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )
        self._loaded[alias] = model
        self._meta[alias] = {
            "alias": alias,
            "path": str(model_path),
            "ctx": ctx,
            "gpu_layers": gpu_layers,
            "loaded_at": time.time(),
        }
        logger.info("Model '%s' loaded", alias)
        return self

    def unload(self, alias):
        """Drop a loaded model. Returns self for chaining."""
        if alias not in self._loaded:
            raise KeyError(f"No model loaded under alias '{alias}'")
        del self._loaded[alias]
        del self._meta[alias]
        logger.info("Model '%s' unloaded", alias)
        return self

    def chat(self, alias, message, *, system=None, temperature=0.7, max_tokens=2048):
        """Send a message, get a plain string back.

        message can be a string or a list of {role, content} dicts.
        """
        model = self._get(alias)
        msgs = self._build_messages(message, system)
        response = model.create_chat_completion(
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["message"]["content"]

    def stream(self, alias, message, *, system=None, temperature=0.7, max_tokens=2048):
        """Send a message, get an iterator of token strings back."""
        model = self._get(alias)
        msgs = self._build_messages(message, system)
        response = model.create_chat_completion(
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in response:
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                yield token

    def session(self, alias, *, system=None, temperature=0.7):
        """Open a conversation session that keeps its own history."""
        self._get(alias)
        return Session(self, alias, system=system, temperature=temperature)

    def serve(self, port=7437, host="localhost", token=None):
        """Start an optional HTTP server. Never required for normal use.

        Exposes OpenAI-compatible /v1/chat/completions plus simple
        model management endpoints. Blocks until stopped.

        If token is provided, all requests must include:
            Authorization: Bearer <token>
        """
        try:
            import uvicorn
            from fastapi import FastAPI, HTTPException, Request
            from fastapi.middleware.cors import CORSMiddleware
            from fastapi.responses import StreamingResponse
        except ImportError as exc:
            raise ImportError(
                "engine.serve() needs fastapi and uvicorn. "
                "Install them with: pip install 'thirthfloor[server]'"
            ) from exc

        import json

        engine = self
        _token = token
        app = FastAPI(title="3th Floor AI Engine", version="1.0.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        def _check_auth(request: Request):
            if not _token:
                return
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != _token:
                raise HTTPException(status_code=401, detail="Unauthorized")

        @app.get("/health")
        def health():
            return {"status": "ok", "loaded": engine.loaded()}

        @app.get("/v1/models")
        def list_models():
            data = []
            for alias in engine.loaded():
                meta = engine._meta[alias]
                data.append(
                    {
                        "id": alias,
                        "object": "model",
                        "created": int(meta["loaded_at"]),
                        "owned_by": "3thfloor",
                    }
                )
            return {"object": "list", "data": data}

        @app.get("/models/list")
        def models_list():
            return {"loaded": [engine._meta[a] for a in engine.loaded()]}

        @app.post("/models/load")
        def models_load(body: dict):
            alias = body.get("alias")
            path = body.get("path")
            if not alias or not path:
                raise HTTPException(status_code=400, detail="alias and path are required")
            try:
                engine.load(
                    alias,
                    path,
                    ctx=int(body.get("ctx", 4096)),
                    gpu_layers=int(body.get("gpu_layers", -1)),
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "loaded", "alias": alias}

        @app.post("/models/unload")
        def models_unload(body: dict):
            alias = body.get("alias")
            if not alias:
                raise HTTPException(status_code=400, detail="alias is required")
            try:
                engine.unload(alias)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "unloaded", "alias": alias}

        @app.post("/v1/chat/completions")
        def chat_completions(request: Request, body: dict):
            _check_auth(request)
            alias = body.get("model")
            messages = body.get("messages")
            if not alias or not messages:
                raise HTTPException(status_code=400, detail="model and messages are required")
            if alias not in engine:
                raise HTTPException(status_code=404, detail=f"Model '{alias}' is not loaded")
            temperature = float(body.get("temperature", 0.7))
            max_tokens = int(body.get("max_tokens", 2048))
            created = int(time.time())
            completion_id = f"chatcmpl-3thfloor-{created}"

            if body.get("stream"):
                def event_stream():
                    for token in engine.stream(
                        alias,
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ):
                        payload = {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": alias,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token},
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(payload)}\n\n"
                    done = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": alias,
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "stop"}
                        ],
                    }
                    yield f"data: {json.dumps(done)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(event_stream(), media_type="text/event-stream")

            tools = body.get("tools")

            if tools:
                # Manual tool calling: inject tools as a compact system prompt and
                # parse the model's JSON response. Bypasses llama-cpp-python's grammar-
                # based tool handling, which inflates context to 90k+ tokens for models
                # without a native tool-calling chat template (e.g. Gemma 4).
                tool_lines = []
                for t in tools:
                    fn = t.get("function", t)
                    name = fn.get("name", "")
                    desc = fn.get("description", "")
                    props = fn.get("parameters", {}).get("properties", {})
                    required = fn.get("parameters", {}).get("required", [])
                    arg_descs = []
                    for pname, pdef in props.items():
                        req_marker = " (required)" if pname in required else ""
                        arg_descs.append(
                            f"    {pname}{req_marker}: {pdef.get('description', pdef.get('type', ''))}"
                        )
                    tool_lines.append(f"- {name}: {desc}")
                    if arg_descs:
                        tool_lines.extend(arg_descs)

                tool_system = (
                    "You have access to the following tools. When you need to call a tool, "
                    "respond ONLY with a JSON object in this exact format and nothing else:\n"
                    '{"tool": "<tool_name>", "arguments": {<arg_key>: <arg_value>, ...}}\n\n'
                    "Available tools:\n" + "\n".join(tool_lines) + "\n\n"
                    "If you do not need to call a tool, respond normally in plain text."
                )

                full_messages = list(messages)
                if full_messages and full_messages[0].get("role") == "system":
                    full_messages[0] = {
                        "role": "system",
                        "content": tool_system + "\n\n" + full_messages[0]["content"],
                    }
                else:
                    full_messages.insert(0, {"role": "system", "content": tool_system})

                content = engine.chat(
                    alias,
                    full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                # Attempt to parse a tool call from the response
                import re as _re
                content_stripped = (content or "").strip()
                try:
                    json_match = _re.search(r'\{.*\}', content_stripped, _re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        if "tool" in parsed and "arguments" in parsed:
                            tool_call_id = f"call_{completion_id}"
                            return {
                                "id": completion_id,
                                "object": "chat.completion",
                                "created": created,
                                "model": alias,
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": None,
                                        "tool_calls": [{
                                            "id": tool_call_id,
                                            "type": "function",
                                            "function": {
                                                "name": parsed["tool"],
                                                "arguments": json.dumps(parsed["arguments"]),
                                            },
                                        }],
                                    },
                                    "finish_reason": "tool_calls",
                                }],
                                "usage": {},
                            }
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

                # No tool call detected — return as normal text
                return {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": created,
                    "model": alias,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }],
                    "usage": {},
                }

            content = engine.chat(
                alias,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": alias,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
            }

        logger.info("Serving on http://%s:%s", host, port)
        uvicorn.run(app, host=host, port=port)

    def loaded(self):
        """Return the list of loaded alias strings."""
        return list(self._loaded.keys())

    def __contains__(self, alias):
        return alias in self._loaded

    def _get(self, alias):
        """Fetch a loaded model or raise a helpful error."""
        if alias not in self._loaded:
            available = ", ".join(self.loaded()) or "none"
            raise KeyError(
                f"No model loaded under alias '{alias}'. Loaded: {available}. "
                f"Use engine.load('{alias}', path) first."
            )
        return self._loaded[alias]

    @staticmethod
    def _build_messages(message, system):
        """Normalize input into a list of chat messages."""
        if isinstance(message, str):
            msgs = [{"role": "user", "content": message}]
            if system:
                msgs.insert(0, {"role": "system", "content": system})
            return msgs
        if isinstance(message, list):
            return message
        raise TypeError(
            "message must be a string or a list of {role, content} dicts, "
            f"got {type(message).__name__}"
        )
