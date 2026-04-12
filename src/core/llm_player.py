"""
API abstraction layer for LLM players.

Provides a unified interface for different LLM providers (OpenAI, Anthropic)
and a human player for interactive play. All players implement the same
generate() method that takes a conversation history and returns a response.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel


class Message(BaseModel):
    """A single message in a conversation."""

    role: str  # "system", "user", or "assistant"
    content: str


@dataclass
class PlayerConfig:
    """Configuration for an LLM player."""

    model: str
    temperature: float = 0.7
    max_tokens: int = 1024
    system_prompt: str = ""


class LLMPlayer(ABC):
    """Abstract base class for all LLM players in the arena."""

    def __init__(self, name: str, config: PlayerConfig) -> None:
        self.name = name
        self.config = config
        self._message_count = 0

    @abstractmethod
    def generate(self, messages: list[Message]) -> str:
        """Generate a response given conversation history.

        Args:
            messages: List of Message objects representing the conversation so far.

        Returns:
            The generated text response.
        """

    @property
    def display_name(self) -> str:
        """Human-readable name including model identifier."""
        return f"{self.name} ({self.config.model})"

    def reset(self) -> None:
        """Reset player state for a new game."""
        self._message_count = 0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, model={self.config.model!r})"


class OpenAIPlayer(LLMPlayer):
    """Player using the OpenAI API (GPT-4, GPT-3.5, etc.)."""

    def __init__(self, name: str, config: PlayerConfig) -> None:
        super().__init__(name, config)
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required for OpenAIPlayer"
            )
        try:
            from openai import OpenAI

            self._client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package is required: pip install openai")

    def generate(self, messages: list[Message]) -> str:
        """Generate a response using the OpenAI API."""
        api_messages = []
        if self.config.system_prompt:
            api_messages.append({"role": "system", "content": self.config.system_prompt})
        for msg in messages:
            api_messages.append({"role": msg.role, "content": msg.content})

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=api_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        self._message_count += 1
        return response.choices[0].message.content or ""


class AnthropicPlayer(LLMPlayer):
    """Player using the Anthropic API (Claude models)."""

    def __init__(self, name: str, config: PlayerConfig) -> None:
        super().__init__(name, config)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable is required for AnthropicPlayer"
            )
        try:
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)
        except ImportError:
            raise ImportError("anthropic package is required: pip install anthropic")

    def generate(self, messages: list[Message]) -> str:
        """Generate a response using the Anthropic API."""
        # Anthropic API uses system parameter separately
        api_messages = []
        for msg in messages:
            if msg.role == "system":
                continue  # handled via system parameter
            role = "user" if msg.role == "user" else "assistant"
            api_messages.append({"role": role, "content": msg.content})

        # Ensure messages alternate user/assistant; Anthropic requires this
        if not api_messages or api_messages[0]["role"] != "user":
            api_messages.insert(0, {"role": "user", "content": "Begin."})

        response = self._client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            system=self.config.system_prompt or "You are a helpful assistant.",
            messages=api_messages,
            temperature=self.config.temperature,
        )
        self._message_count += 1
        return response.content[0].text


class HumanPlayer(LLMPlayer):
    """Interactive human player for human-in-the-loop games."""

    def __init__(self, name: str = "Human") -> None:
        config = PlayerConfig(model="human", temperature=0.0, max_tokens=0)
        super().__init__(name, config)
        self._input_fn = input  # overridable for testing / Streamlit

    def generate(self, messages: list[Message]) -> str:
        """Prompt the human for input."""
        if messages:
            last = messages[-1]
            print(f"\n[{last.role}]: {last.content}")
        print()
        response = self._input_fn(f"[{self.name}] Your response: ")
        self._message_count += 1
        return response


class MockPlayer(LLMPlayer):
    """Deterministic mock player for testing."""

    def __init__(
        self,
        name: str = "Mock",
        responses: list[str] | None = None,
        config: PlayerConfig | None = None,
    ) -> None:
        if config is None:
            config = PlayerConfig(model="mock-v1")
        super().__init__(name, config)
        self._responses = responses or ["Mock response."]
        self._call_index = 0

    def generate(self, messages: list[Message]) -> str:
        """Return the next pre-configured response."""
        response = self._responses[self._call_index % len(self._responses)]
        self._call_index += 1
        self._message_count += 1
        return response

    def reset(self) -> None:
        super().reset()
        self._call_index = 0


def create_player(
    provider: str,
    name: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    system_prompt: str = "",
) -> LLMPlayer:
    """Factory function to create an LLM player.

    Args:
        provider: One of "openai", "anthropic", "human", "mock".
        name: Display name for the player.
        model: Model identifier (e.g., "gpt-4", "claude-3-opus-20240229").
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.
        system_prompt: System prompt / persona.

    Returns:
        An LLMPlayer instance.
    """
    config = PlayerConfig(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
    )
    providers = {
        "openai": OpenAIPlayer,
        "anthropic": AnthropicPlayer,
        "human": lambda n, c: HumanPlayer(n),
        "mock": lambda n, c: MockPlayer(n, config=c),
    }
    factory = providers.get(provider.lower())
    if factory is None:
        raise ValueError(f"Unknown provider: {provider}. Choose from {list(providers.keys())}")
    return factory(name, config)
