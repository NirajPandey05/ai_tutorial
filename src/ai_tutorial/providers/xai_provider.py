"""
xAI (Grok) Provider Implementation.

Supports Grok models via the OpenAI-compatible API.
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


class XAIProvider(LLMProvider):
    """xAI (Grok) API provider using OpenAI-compatible API."""
    
    name = "xai"
    display_name = "xAI (Grok)"
    default_model = "grok-4-mini"
    available_models = [
        "grok-4",
        "grok-4-mini",
        "grok-3",
        "grok-3-mini",
        "grok-3-vision",
    ]
    supports_streaming = True
    supports_system_message = True
    
    XAI_BASE_URL = "https://api.x.ai/v1"
    
    def _setup_client(self) -> None:
        """Set up the xAI client using OpenAI SDK."""
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or self.XAI_BASE_URL,
        )
    
    def _convert_messages(self, messages: List[LLMMessage]) -> List[Dict[str, str]]:
        """Convert messages to OpenAI-compatible format."""
        return [msg.to_dict() for msg in messages]
    
    async def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send a chat completion request to xAI."""
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
        """Stream a chat completion from xAI."""
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
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(
                        content=chunk.choices[0].delta.content,
                        is_final=False,
                    )
                
                # Check for final chunk
                if chunk.choices and chunk.choices[0].finish_reason:
                    usage = {}
                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = {
                            "input_tokens": chunk.usage.prompt_tokens,
                            "output_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }
                    yield StreamChunk(
                        content="",
                        is_final=True,
                        finish_reason=chunk.choices[0].finish_reason,
                        usage=usage,
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
        """Validate the xAI API key."""
        try:
            # Make a minimal API call to validate the key
            response = await self._client.models.list()
            models = [m.id for m in response.data[:3]]
            return True, f"API key valid. Available models: {', '.join(models)}"
        except OpenAIAuthError:
            return False, "Invalid API key. Please check your xAI API key."
        except Exception as e:
            # xAI might not support model listing, try a simple completion
            try:
                test_response = await self._client.chat.completions.create(
                    model=self.default_model,
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=5,
                )
                return True, f"API key valid. Using model: {test_response.model}"
            except OpenAIAuthError:
                return False, "Invalid API key. Please check your xAI API key."
            except Exception as e2:
                return False, f"Error validating key: {str(e2)}"
