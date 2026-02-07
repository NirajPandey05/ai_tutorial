"""
Tests for LLM Provider Adapters

Tests cover:
- Provider initialization
- Message formatting
- Response parsing
- Error handling
- API key validation (mocked)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai_tutorial.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMMessage,
    ProviderConfig,
    MessageRole,
    StreamChunk,
)
from ai_tutorial.providers.factory import ProviderFactory, get_provider


class TestLLMMessage:
    """Tests for LLMMessage dataclass."""
    
    def test_create_user_message(self):
        msg = LLMMessage(role=MessageRole.USER, content="Hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "Hello"
    
    def test_create_system_message(self):
        msg = LLMMessage(role=MessageRole.SYSTEM, content="You are helpful")
        assert msg.role == MessageRole.SYSTEM
        assert msg.content == "You are helpful"
    
    def test_create_assistant_message(self):
        msg = LLMMessage(role=MessageRole.ASSISTANT, content="Hi there!")
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "Hi there!"
    
    def test_to_dict(self):
        msg = LLMMessage(role=MessageRole.USER, content="Test")
        result = msg.to_dict()
        assert result == {"role": "user", "content": "Test"}


class TestProviderConfig:
    """Tests for ProviderConfig dataclass."""
    
    def test_minimal_config(self):
        config = ProviderConfig(api_key="test-key")
        assert config.api_key == "test-key"
        assert config.temperature == 0.7
        assert config.max_tokens == 1024
    
    def test_full_config(self):
        config = ProviderConfig(
            api_key="test-key",
            model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            top_p=0.9,
            base_url="https://custom.api.com",
            extra_params={"custom": "value"}
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.base_url == "https://custom.api.com"
        assert config.extra_params["custom"] == "value"


class TestLLMResponse:
    """Tests for LLMResponse dataclass."""
    
    def test_basic_response(self):
        response = LLMResponse(
            content="Hello!",
            model="gpt-4",
            provider="openai"
        )
        assert response.content == "Hello!"
        assert response.model == "gpt-4"
        assert response.provider == "openai"
    
    def test_response_with_usage(self):
        response = LLMResponse(
            content="Response",
            model="gpt-4",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        )
        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_tokens == 30
    
    def test_response_with_anthropic_usage(self):
        """Test that Anthropic-style usage fields work."""
        response = LLMResponse(
            content="Response",
            model="claude-3",
            provider="anthropic",
            usage={"input_tokens": 15, "output_tokens": 25}
        )
        assert response.input_tokens == 15
        assert response.output_tokens == 25
        assert response.total_tokens == 40


class TestOpenAIProvider:
    """Tests for OpenAI provider."""
    
    @pytest.fixture
    def config(self):
        return ProviderConfig(api_key="test-openai-key", model="gpt-4o-mini")
    
    def test_provider_metadata(self, config):
        with patch('openai.AsyncOpenAI'):
            from ai_tutorial.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(config)
            assert provider.name == "openai"
            assert provider.display_name == "OpenAI"
            assert "gpt-4o" in provider.available_models
            assert provider.supports_streaming is True
    
    def test_default_model(self, config):
        with patch('openai.AsyncOpenAI'):
            from ai_tutorial.providers.openai_provider import OpenAIProvider
            config.model = ""
            provider = OpenAIProvider(config)
            assert provider.get_model() == provider.default_model
    
    def test_convert_messages(self, config):
        with patch('openai.AsyncOpenAI'):
            from ai_tutorial.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(config)
            messages = [
                LLMMessage(MessageRole.SYSTEM, "Be helpful"),
                LLMMessage(MessageRole.USER, "Hi")
            ]
            formatted = provider._convert_messages(messages)
            assert len(formatted) == 2
            assert formatted[0]["role"] == "system"
            assert formatted[1]["role"] == "user"


class TestAnthropicProvider:
    """Tests for Anthropic provider."""
    
    @pytest.fixture
    def config(self):
        return ProviderConfig(api_key="test-anthropic-key", model="claude-3-5-sonnet-20241022")
    
    def test_provider_metadata(self, config):
        with patch('anthropic.AsyncAnthropic'):
            from ai_tutorial.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(config)
            assert provider.name == "anthropic"
            assert "Anthropic" in provider.display_name
            assert len(provider.available_models) > 0
    
    def test_system_message_extraction(self, config):
        """Anthropic handles system messages separately."""
        with patch('anthropic.AsyncAnthropic'):
            from ai_tutorial.providers.anthropic_provider import AnthropicProvider
            provider = AnthropicProvider(config)
            messages = [
                LLMMessage(MessageRole.SYSTEM, "Be concise"),
                LLMMessage(MessageRole.USER, "Hello")
            ]
            system_msg, formatted = provider._convert_messages(messages)
            assert system_msg == "Be concise"
            assert len(formatted) == 1
            assert formatted[0]["role"] == "user"


class TestGoogleProvider:
    """Tests for Google Gemini provider."""
    
    @pytest.fixture
    def config(self):
        return ProviderConfig(api_key="test-google-key", model="gemini-1.5-flash")
    
    def test_provider_metadata(self, config):
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                from ai_tutorial.providers.google_provider import GoogleProvider
                provider = GoogleProvider(config)
                assert provider.name == "google"
                assert "Google" in provider.display_name
                assert len(provider.available_models) > 0


class TestXAIProvider:
    """Tests for xAI Grok provider."""
    
    @pytest.fixture
    def config(self):
        return ProviderConfig(api_key="test-xai-key", model="grok-3")
    
    def test_provider_metadata(self, config):
        with patch('openai.AsyncOpenAI'):
            from ai_tutorial.providers.xai_provider import XAIProvider
            provider = XAIProvider(config)
            assert provider.name == "xai"
            assert "Grok" in provider.display_name
            assert len(provider.available_models) > 0


class TestProviderFactory:
    """Tests for the provider factory."""
    
    def test_get_openai_provider(self):
        with patch('openai.AsyncOpenAI'):
            from ai_tutorial.providers.openai_provider import OpenAIProvider
            provider = ProviderFactory.create(
                "openai",
                ProviderConfig(api_key="test-key")
            )
            assert isinstance(provider, OpenAIProvider)
    
    def test_get_anthropic_provider(self):
        with patch('anthropic.AsyncAnthropic'):
            from ai_tutorial.providers.anthropic_provider import AnthropicProvider
            provider = ProviderFactory.create(
                "anthropic",
                ProviderConfig(api_key="test-key")
            )
            assert isinstance(provider, AnthropicProvider)
    
    def test_get_google_provider(self):
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                from ai_tutorial.providers.google_provider import GoogleProvider
                provider = ProviderFactory.create(
                    "google",
                    ProviderConfig(api_key="test-key")
                )
                assert isinstance(provider, GoogleProvider)
    
    def test_get_xai_provider(self):
        with patch('openai.AsyncOpenAI'):
            from ai_tutorial.providers.xai_provider import XAIProvider
            provider = ProviderFactory.create(
                "xai",
                ProviderConfig(api_key="test-key")
            )
            assert isinstance(provider, XAIProvider)
    
    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.create(
                "unknown",
                ProviderConfig(api_key="test-key")
            )
    
    def test_list_providers(self):
        providers = ProviderFactory.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "xai" in providers


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""
    
    def test_content_chunk(self):
        chunk = StreamChunk(content="Hello")
        assert chunk.content == "Hello"
        assert chunk.is_final is False
    
    def test_final_chunk(self):
        chunk = StreamChunk(content="", is_final=True, finish_reason="stop")
        assert chunk.is_final is True
        assert chunk.finish_reason == "stop"


@pytest.mark.asyncio
class TestAsyncProviderMethods:
    """Async tests for provider methods."""
    
    async def test_openai_chat_mock(self):
        """Test that OpenAI provider can be instantiated with mocking."""
        with patch('openai.AsyncOpenAI') as mock_client:
            from ai_tutorial.providers.openai_provider import OpenAIProvider
            provider = OpenAIProvider(ProviderConfig(api_key="test-key"))
            
            # Verify provider was created
            assert provider.name == "openai"
            assert provider._client is not None

