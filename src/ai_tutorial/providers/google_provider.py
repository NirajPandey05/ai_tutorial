"""
Google Gemini Provider Implementation.

Supports Gemini Pro, Gemini Ultra, and other Google AI models.
"""

from typing import AsyncGenerator, List, Optional, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import exceptions as google_exceptions

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


class GoogleProvider(LLMProvider):
    """Google Gemini API provider."""
    
    name = "google"
    display_name = "Google (Gemini)"
    default_model = "gemini-3-flash"
    available_models = [
        "gemini-3-ultra",
        "gemini-3-pro",
        "gemini-3-flash",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2-flash",
    ]
    supports_streaming = True
    supports_system_message = True
    
    def _setup_client(self) -> None:
        """Set up the Google Generative AI client."""
        genai.configure(api_key=self.config.api_key)
        self._client = genai
    
    def _convert_messages(self, messages: List[LLMMessage]) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Convert messages to Gemini format.
        
        Returns (system_instruction, conversation_history)
        """
        system_instruction = None
        history = []
        
        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_instruction = msg.content
            elif msg.role == MessageRole.USER:
                history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == MessageRole.ASSISTANT:
                history.append({"role": "model", "parts": [msg.content]})
        
        return system_instruction, history
    
    def _get_generation_config(self, params: Dict[str, Any]) -> genai.GenerationConfig:
        """Create generation config from parameters."""
        return genai.GenerationConfig(
            temperature=params.get("temperature", 0.7),
            max_output_tokens=params.get("max_tokens", 1024),
            top_p=params.get("top_p", 1.0),
        )
    
    def _get_safety_settings(self) -> Dict[HarmCategory, HarmBlockThreshold]:
        """Get default safety settings for educational content.
        
        Uses BLOCK_MEDIUM_AND_ABOVE to allow most educational content
        while still filtering clearly harmful content.
        """
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
    
    async def chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """Send a chat completion request to Google Gemini."""
        try:
            params = self._merge_params(**kwargs)
            model_name = self.get_model(model)
            system_instruction, history = self._convert_messages(messages)
            
            # Create model with system instruction
            model_kwargs = {}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            
            gemini_model = self._client.GenerativeModel(
                model_name=model_name,
                generation_config=self._get_generation_config(params),
                safety_settings=self._get_safety_settings(),
                **model_kwargs,
            )
            
            # Get the last user message and history
            if history:
                last_message = history[-1]["parts"][0]
                chat_history = history[:-1] if len(history) > 1 else []
                
                # Start chat with history
                chat = gemini_model.start_chat(history=chat_history)
                response = await chat.send_message_async(last_message)
            else:
                response = await gemini_model.generate_content_async("Hello")
            
            # Extract usage
            usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                usage = {
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count,
                }
            
            return LLMResponse(
                content=response.text,
                model=model_name,
                provider=self.name,
                usage=usage,
                finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                raw_response=response,
            )
            
        except google_exceptions.InvalidArgument as e:
            if "API_KEY" in str(e) or "api key" in str(e).lower():
                raise AuthenticationError(
                    "Invalid API key", self.name, original_error=e
                )
            raise ProviderError(
                f"Invalid argument: {str(e)}", self.name, original_error=e
            )
        except google_exceptions.ResourceExhausted as e:
            raise RateLimitError(
                "Rate limit exceeded", self.name, original_error=e
            )
        except Exception as e:
            if "API_KEY" in str(e) or "api key" in str(e).lower():
                raise AuthenticationError(
                    "Invalid API key", self.name, original_error=e
                )
            raise ProviderError(
                f"API error: {str(e)}", self.name, original_error=e
            )
    
    async def stream_chat(
        self,
        messages: List[LLMMessage],
        model: Optional[str] = None,
        **kwargs
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream a chat completion from Google Gemini."""
        try:
            params = self._merge_params(**kwargs)
            model_name = self.get_model(model)
            system_instruction, history = self._convert_messages(messages)
            
            # Create model with system instruction
            model_kwargs = {}
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction
            
            gemini_model = self._client.GenerativeModel(
                model_name=model_name,
                generation_config=self._get_generation_config(params),
                safety_settings=self._get_safety_settings(),
                **model_kwargs,
            )
            
            # Get the last user message and history
            if history:
                last_message = history[-1]["parts"][0]
                chat_history = history[:-1] if len(history) > 1 else []
                
                # Start chat with history
                chat = gemini_model.start_chat(history=chat_history)
                response = await chat.send_message_async(last_message, stream=True)
            else:
                response = await gemini_model.generate_content_async("Hello", stream=True)
            
            usage = {}
            async for chunk in response:
                if chunk.text:
                    yield StreamChunk(content=chunk.text, is_final=False)
                
                # Capture usage from final chunk
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    usage = {
                        "input_tokens": chunk.usage_metadata.prompt_token_count,
                        "output_tokens": chunk.usage_metadata.candidates_token_count,
                        "total_tokens": chunk.usage_metadata.total_token_count,
                    }
            
            # Final chunk
            yield StreamChunk(
                content="",
                is_final=True,
                finish_reason="stop",
                usage=usage,
            )
                    
        except google_exceptions.InvalidArgument as e:
            if "API_KEY" in str(e) or "api key" in str(e).lower():
                raise AuthenticationError(
                    "Invalid API key", self.name, original_error=e
                )
            raise ProviderError(
                f"Invalid argument: {str(e)}", self.name, original_error=e
            )
        except google_exceptions.ResourceExhausted as e:
            raise RateLimitError(
                "Rate limit exceeded", self.name, original_error=e
            )
        except Exception as e:
            if "API_KEY" in str(e) or "api key" in str(e).lower():
                raise AuthenticationError(
                    "Invalid API key", self.name, original_error=e
                )
            raise ProviderError(
                f"API error: {str(e)}", self.name, original_error=e
            )
    
    async def validate_api_key(self) -> tuple[bool, str]:
        """Validate the Google API key."""
        try:
            # List models to validate the key
            models = []
            for model in self._client.list_models():
                if "generateContent" in model.supported_generation_methods:
                    models.append(model.name.replace("models/", ""))
                if len(models) >= 3:
                    break
            
            return True, f"API key valid. Available models include: {', '.join(models)}"
        except Exception as e:
            if "API_KEY" in str(e) or "api key" in str(e).lower():
                return False, "Invalid API key. Please check your Google AI API key."
            return False, f"Error validating key: {str(e)}"
