"""
session.py - Stateful conversation session for the 3th Floor AI SDK.

A Session wraps an engine alias and maintains message history automatically.
Calling send() or stream() appends the user turn, invokes the model, then
appends the assistant reply before returning. The developer never touches
the message list directly.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Generator, Iterator, List, Optional

if TYPE_CHECKING:
    from .engine import Engine


class Session:
    """
    Maintains a running conversation with one model alias.

    Usage:
        session = engine.session("mistral-7b", system="You are a code reviewer.")
        reply = session.send("What is wrong with this function?")
        for token in session.stream("Give me a one-line fix."):
            print(token, end="", flush=True)
        print()
        session.clear()
    """

    def __init__(
        self,
        engine: "Engine",
        alias: str,
        *,
        system: Optional[str] = None,
        temperature: float = 0.7,
    ) -> None:
        self._engine = engine
        self._alias = alias
        self._temperature = temperature
        self._messages: List[dict] = []
        if system:
            self._messages.append({"role": "system", "content": system})

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def send(self, message: str, *, max_tokens: int = 2048) -> str:
        """
        Send a user message and return the assistant reply as a plain string.

        The user turn and assistant reply are both stored in history so
        subsequent calls carry full context.
        """
        self._messages.append({"role": "user", "content": message})
        response: str = self._engine.chat(
            self._alias,
            self._messages,
            temperature=self._temperature,
            max_tokens=max_tokens,
        )
        self._messages.append({"role": "assistant", "content": response})
        return response

    def stream(
        self, message: str, *, max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        """
        Send a user message and yield tokens one at a time.

        The full reply is accumulated internally and appended to history
        only after the iterator is exhausted. This means history is
        consistent whether the caller drains the iterator or abandons it
        mid-stream (partial replies are NOT stored on early exit).
        """
        self._messages.append({"role": "user", "content": message})
        full_response = ""
        for token in self._engine.stream(
            self._alias,
            self._messages,
            temperature=self._temperature,
            max_tokens=max_tokens,
        ):
            full_response += token
            yield token
        self._messages.append({"role": "assistant", "content": full_response})

    def clear(self) -> "Session":
        """
        Clear conversation history while keeping the system prompt intact.

        Returns self so calls can be chained:
            session.clear().send("Start over.")
        """
        if self._messages and self._messages[0]["role"] == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def history(self) -> List[dict]:
        """
        User and assistant turns in order. System prompt is excluded.

        Each entry is {"role": "user"|"assistant", "content": "..."}.
        The list is a copy; mutating it does not affect the session.
        """
        return [m for m in self._messages if m["role"] != "system"]

    @property
    def system(self) -> Optional[str]:
        """
        The system prompt string passed at construction, or None if not set.
        """
        if self._messages and self._messages[0]["role"] == "system":
            return self._messages[0]["content"]
        return None
