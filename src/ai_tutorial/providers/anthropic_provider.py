"""
Anthropic Provider Implementation.

Supports Claude 3.5, Claude 3, and other Anthropic models.
"""

from typing import AsyncGenerator, List, Optional, Dict, Any
from anthropic import AsyncAnthropic, APIError, AuthenticationError as AnthropicAuthError, RateLimitError as AnthropicRateLimitError

from .base import (
    LLMProvider,
    LLMMessage,
    LLMResponse,
    StreamChunk,
    ProviderConfig,
    MessageRole,
    AuthenticationError,
    RateLimitError,
    ProviderError,
)


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) API provider."""
    
    name = "anthropic"
    display_name = "Anthropic (Claude)"
    default_model = "claude-4.5-sonnet"
    available_models = [
        "claude-4.5-opus",
        "claude-4.5-sonnet",
        "claude-4.5-haiku",
        "claude-4-opus",
        "claude-4-sonnet",
        "claude-3.5-sonnet",
        "claude-3.5-haiku",
    ]
    supports_streaming = True
    supports_system_message = True
    
    def _setup_client(self) -> None:
        """Set up the Anthropic client."""
        self._client = AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
    
    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[Optional[str], List[Dict[str, str]]]:
        """
        Convert messages to Anthropic format.
        
        Anthropic requires system message to be separate.
        Returns (system_message, conversation_messages)
        """
        system_message = None
        conversation = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_message = msg.content
            else:
                conversation.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })
        
        return system_message, conversation
    
    async def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send a chat completion request to Anthropic."""
        try:
            params = self._merge_params(**kwargs)
            model_name = self.get_model(model)
            system_message, conversation = self._convert_messages(messages)
            
            # Build request kwargs
            request_kwargs = {
                "model": model_name,
                "messages": conversation,
                "max_tokens": params.get("max_tokens", 1024),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0),
            }
            
            if system_message:
                request_kwargs["system"] = system_message
            
            response = await self._client.messages.create(**request_kwargs)
            
            # Extract content from response
            content = ""
            if response.content:
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text
            
            return LLMResponse(
                content=content,
                model=response.model,
                provider=self.name,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
                },
                finish_reason=response.stop_reason,
                raw_response=response,
            )
            
        except AnthropicAuthError as e:
            raise AuthenticationError(
                "Invalid API key", self.name, original_error=e
            )
        except AnthropicRateLimitError as e:
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
        """Stream a chat completion from Anthropic."""
        try:
            params = self._merge_params(**kwargs)
            model_name = self.get_model(model)
            system_message, conversation = self._convert_messages(messages)
            
            # Build request kwargs
            request_kwargs = {
                "model": model_name,
                "messages": conversation,
                "max_tokens": params.get("max_tokens", 1024),
                "temperature": params.get("temperature", 0.7),
                "top_p": params.get("top_p", 1.0),
            }
            
            if system_message:
                request_kwargs["system"] = system_message
            
            async with self._client.messages.stream(**request_kwargs) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(content=text, is_final=False)
                
                # Get final message for usage
                final_message = await stream.get_final_message()
                yield StreamChunk(
                    content="",
                    is_final=True,
                    finish_reason=final_message.stop_reason,
                    usage={
                        "input_tokens": final_message.usage.input_tokens,
                        "output_tokens": final_message.usage.output_tokens,
                        "total_tokens": final_message.usage.input_tokens + final_message.usage.output_tokens,
                    },
                )
                    
        except AnthropicAuthError as e:
            raise AuthenticationError(
                "Invalid API key", self.name, original_error=e
            )
        except AnthropicRateLimitError as e:
            raise RateLimitError(
                "Rate limit exceeded", self.name, original_error=e
            )
        except APIError as e:
            raise ProviderError(
                f"API error: {str(e)}", self.name, original_error=e
            )
    
    async def validate_api_key(self) -> tuple[bool, str]:
        """Validate the Anthropic API key."""
        try:
            # Make a minimal API call to validate the key
            response = await self._client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            return True, f"API key valid. Using model: {response.model}"
        except AnthropicAuthError:
            return False, "Invalid API key. Please check your Anthropic API key."
        except Exception as e:
            return False, f"Error validating key: {str(e)}"
