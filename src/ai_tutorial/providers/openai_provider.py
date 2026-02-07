"""
OpenAI Provider Implementation.

Supports GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, and other OpenAI models.
"""

from typing import AsyncGenerator, List, Optional, Dict, Any
from openai import AsyncOpenAI, APIError, AuthenticationError as OpenAIAuthError, RateLimitError as OpenAIRateLimitError

from .base import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    ProviderConfig,
    AuthenticationError,
    RateLimitError,
    ProviderError,
)


class OpenAIProvider(LLMProvider):
    """OpenAI API provider."""
    
    name = "openai"
    display_name = "OpenAI"
    default_model = "gpt-5.2-mini"
    available_models = [
        "gpt-5.2",
        "gpt-5.2-mini",
        "gpt-5.1",
        "gpt-5.1-mini",
        "gpt-5",
        "gpt-4o",
        "gpt-4o-mini",
        "o3",
        "o3-mini",
        "o1",
    ]
    supports_streaming = True
    supports_system_message = True
    
    def _setup_client(self) -> None:
        """Set up the OpenAI client."""
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
    
    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert messages to OpenAI format."""
        return [msg.to_dict() for msg in messages]
    
    async def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send a chat completion request to OpenAI."""
        try:
            params = self._merge_params(**kwargs)
            model_name = self.get_model(model)
            
            response = await self._client.chat.completions.create(
                model=model_name,
                messages=self._convert_messages(messages),
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1024),
                top_p=params.get("top_p", 1.0),
                stream=False,
            )
            
            choice = response.choices[0]
            return LLMResponse(
                content=choice.message.content or "",
                model=response.model,
                provider=self.name,
                usage={
                    "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "output_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                finish_reason=choice.finish_reason,
                raw_response=response,
            )
            
        except OpenAIAuthError as e:
            raise AuthenticationError(
                "Invalid API key", self.name, original_error=e
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                "Rate limit exceeded", self.name, original_error=e
            )
        except APIError as e:
            raise ProviderError(
                f"API error: {str(e)}", self.name, original_error=e
            )
    
    async def stream_chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion from OpenAI."""
        try:
            params = self._merge_params(**kwargs)
            model_name = self.get_model(model)
            
            stream = await self._client.chat.completions.create(
                model=model_name,
                messages=self._convert_messages(messages),
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 1024),
                top_p=params.get("top_p", 1.0),
                stream=True,
                stream_options={"include_usage": True},
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=False,
                    )
                
                # Final chunk with usage
                if chunk.usage:
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        finish_reason=chunk.choices[0].finish_reason if chunk.choices else None,
                        usage={
                            "input_tokens": chunk.usage.prompt_tokens,
                            "output_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        },
                    )
                elif chunk.choices and chunk.choices[0].finish_reason:
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        finish_reason=chunk.choices[0].finish_reason,
                    )
                    
        except OpenAIAuthError as e:
            raise AuthenticationError(
                "Invalid API key", self.name, original_error=e
            )
        except OpenAIRateLimitError as e:
            raise RateLimitError(
                "Rate limit exceeded", self.name, original_error=e
            )
        except APIError as e:
            raise ProviderError(
                f"API error: {str(e)}", self.name, original_error=e
            )
    
    async def validate_api_key(self) -> tuple[bool, str]:
        """Validate the OpenAI API key."""
        try:
            # Make a minimal API call to validate the key
            response = await self._client.models.list()
            models = [m.id for m in response.data[:5]]
            return True, f"API key valid. Available models include: {', '.join(models)}"
        except OpenAIAuthError:
            return False, "Invalid API key. Please check your OpenAI API key."
        except Exception as e:
            return False, f"Error validating key: {str(e)}"
