"""
Base classes and protocols for LLM providers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional, List, Dict, Any


class MessageRole(str, Enum):
    """Role of a message in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return {"role": self.role.value, "content": self.content}


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider."""
    api_key: str
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    base_url: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    content: str
    model: str
    provider: str
    usage: Optional[Dict[str, int]] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Any] = None
    
    @property
    def input_tokens(self) -> int:
        """Get input token count."""
        if self.usage:
            return self.usage.get("input_tokens", self.usage.get("prompt_tokens", 0))
        return 0
    
    @property
    def output_tokens(self) -> int:
        """Get output token count."""
        if self.usage:
            return self.usage.get("output_tokens", self.usage.get("completion_tokens", 0))
        return 0
    
    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        if self.usage:
            return self.usage.get("total_tokens", self.input_tokens + self.output_tokens)
        return self.input_tokens + self.output_tokens


@dataclass
class StreamChunk:
    """A chunk of streaming response."""
    content: str
    is_final: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    # Provider metadata
    name: str = "base"
    display_name: str = "Base Provider"
    default_model: str = ""
    available_models: List[str] = []
    supports_streaming: bool = True
    supports_system_message: bool = True
    
    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration."""
        self.config = config
        self._client: Any = None
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self) -> None:
        """Set up the provider-specific client."""
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Send a chat completion request.
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional provider-specific parameters
            
        Returns:
            LLMResponse with the completion
        """
        pass
    
    @abstractmethod
    async def stream_chat(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream a chat completion request.
        
        Args:
            messages: List of messages in the conversation
            **kwargs: Additional provider-specific parameters
            
        Yields:
            StreamChunk objects with content deltas
        """
        pass
    
    @abstractmethod
    async def validate_api_key(self) -> tuple[bool, str]:
        """
        Validate the API key by making a minimal API call.
        
        Returns:
            Tuple of (is_valid, message)
        """
        pass
    
    def get_model(self, model: Optional[str] = None) -> str:
        """Get the model to use, with fallback to default."""
        return model or self.config.model or self.default_model
    
    def _merge_params(self, **kwargs) -> Dict[str, Any]:
        """Merge config params with call-specific params."""
        params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        params.update(self.config.extra_params)
        params.update(kwargs)
        return params


class ProviderError(Exception):
    """Base exception for provider errors."""
    
    def __init__(self, message: str, provider: str, original_error: Optional[Exception] = None):
        self.message = message
        self.provider = provider
        self.original_error = original_error
        super().__init__(f"[{provider}] {message}")


class AuthenticationError(ProviderError):
    """Raised when API key is invalid."""
    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, provider: str, retry_after: Optional[float] = None, **kwargs):
        super().__init__(message, provider, **kwargs)
        self.retry_after = retry_after


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not available."""
    pass


class ContextLengthError(ProviderError):
    """Raised when the context length is exceeded."""
    pass
