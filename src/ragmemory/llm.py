import os
from dataclasses import dataclass

NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
DEFAULT_LLM_PROVIDER = "nvidia"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
DEFAULT_OPENCODE_GO_MODEL = "deepseek-v4-flash"
LLM_API_STYLE_OPENAI_CHAT = "openai_chat"


def provider_env_prefix(provider: str) -> str:
    provider_key = "".join(
        ch if ch.isalnum() else "_" for ch in provider.strip().upper()
    ).strip("_")
    return f"RAGMEMORY_LLM_{provider_key}"


@dataclass
class LLMProviderOptions:
    provider: str = DEFAULT_LLM_PROVIDER
    api_key: str | None = None
    base_url: str = DEFAULT_NVIDIA_BASE_URL
    model: str = ""
    api_style: str = LLM_API_STYLE_OPENAI_CHAT
    extra_body: dict | None = None

    @classmethod
    def from_env(cls, provider: str, fallback_model: str) -> "LLMProviderOptions":
        provider = (provider or DEFAULT_LLM_PROVIDER).strip().lower()
        prefix = provider_env_prefix(provider)
        api_key = os.environ.get(f"{prefix}_API_KEY")
        base_url = os.environ.get(f"{prefix}_BASE_URL")
        model = os.environ.get(f"{prefix}_MODEL")
        api_style = os.environ.get(f"{prefix}_API_STYLE", LLM_API_STYLE_OPENAI_CHAT)
        thinking = os.environ.get(f"{prefix}_THINKING", "").strip().lower()

        if provider == "nvidia":
            api_key = api_key or os.environ.get(NVIDIA_API_KEY_ENV)
            base_url = base_url or DEFAULT_NVIDIA_BASE_URL
            model = model or fallback_model
        elif provider == "opencode_go":
            base_url = base_url or DEFAULT_OPENCODE_GO_BASE_URL
            model = model or DEFAULT_OPENCODE_GO_MODEL
        else:
            base_url = base_url or ""
            model = model or fallback_model

        return cls(
            provider=provider,
            api_key=api_key.strip() if api_key else None,
            base_url=base_url.rstrip("/"),
            model=model.strip(),
            api_style=api_style.strip().lower(),
            extra_body=cls._extra_body_for(provider, thinking),
        )

    @staticmethod
    def _extra_body_for(provider: str, thinking: str) -> dict | None:
        if provider != "opencode_go" or not thinking:
            return None
        if thinking == "disabled":
            return {"thinking": {"type": "disabled"}}
        if thinking == "enabled":
            return {"thinking": {"type": "enabled"}}
        return None


class LLMProviderClient:
    def __init__(self, options: LLMProviderOptions):
        self.options = options
        self.client = None
        self.last_error: str | None = None

    def _client(self):
        if self.client:
            return self.client
        if not self.options.api_key:
            self.last_error = f"{self.options.provider} API key missing"
            return None
        if not self.options.base_url:
            self.last_error = f"{self.options.provider} base_url missing"
            return None
        from openai import OpenAI

        self.client = OpenAI(
            base_url=self.options.base_url,
            api_key=self.options.api_key,
        )
        return self.client

    def complete_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0,
    ) -> str | None:
        self.last_error = None
        if self.options.api_style != LLM_API_STYLE_OPENAI_CHAT:
            self.last_error = f"unsupported api_style: {self.options.api_style}"
            return None
        client = self._client()
        if client is None:
            return None
        try:
            response = client.chat.completions.create(
                model=self.options.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self.options.extra_body,
            )
        except Exception as exc:
            self.last_error = str(exc)
            return None
        return (response.choices[0].message.content or "").strip()
