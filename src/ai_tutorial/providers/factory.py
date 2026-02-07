"""
Provider Factory - Creates LLM providers by name.
"""

from typing import Dict, Type, Optional
from .base import LLMProvider, ProviderConfig, ProviderError
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .xai_provider import XAIProvider


class ProviderFactory:
    """Factory for creating LLM provider instances."""
    
    # Registry of available providers
    _providers: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "xai": XAIProvider,
    }
    
    @classmethod
    def register(cls, name: str, provider_class: Type[LLMProvider]) -> None:
        """
        Register a new provider.
        
        Args:
            name: Provider identifier (lowercase)
            provider_class: Provider class to register
        """
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def get_provider_class(cls, name: str) -> Type[LLMProvider]:
        """
        Get a provider class by name.
        
        Args:
            name: Provider identifier
            
        Returns:
            Provider class
            
        Raises:
            ValueError: If provider not found
        """
        name = name.lower()
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys())
            raise ValueError(f"Unknown provider: {name}. Available: {available}")
        return cls._providers[name]
    
    @classmethod
    def create(
        cls,
        name: str,
        api_key: str,
        model: Optional[str] = None,
        **kwargs
    ) -> LLMProvider:
        """
        Create a provider instance.
        
        Args:
            name: Provider identifier
            api_key: API key for the provider
            model: Model to use (optional, uses default if not specified)
            **kwargs: Additional configuration options
            
        Returns:
            Configured provider instance
        """
        provider_class = cls.get_provider_class(name)
        config = ProviderConfig(
            api_key=api_key,
            model=model or "",
            **kwargs
        )
        return provider_class(config)
    
    @classmethod
    def list_providers(cls) -> Dict[str, dict]:
        """
        List all available providers with their metadata.
        
        Returns:
            Dictionary of provider info
        """
        providers = {}
        for name, provider_class in cls._providers.items():
            providers[name] = {
                "name": provider_class.name,
                "display_name": provider_class.display_name,
                "default_model": provider_class.default_model,
                "available_models": provider_class.available_models,
                "supports_streaming": provider_class.supports_streaming,
                "supports_system_message": provider_class.supports_system_message,
            }
        return providers
    
    @classmethod
    def get_models(cls, provider_name: str) -> list:
        """
        Get available models for a provider.
        
        Args:
            provider_name: Provider identifier
            
        Returns:
            List of model names
        """
        provider_class = cls.get_provider_class(provider_name)
        return provider_class.available_models


def get_provider(
    name: str,
    api_key: str,
    model: Optional[str] = None,
    **kwargs
) -> LLMProvider:
    """
    Convenience function to create a provider.
    
    Args:
        name: Provider identifier (openai, anthropic, google, xai)
        api_key: API key for the provider
        model: Model to use (optional)
        **kwargs: Additional configuration
        
    Returns:
        Configured provider instance
    """
    return ProviderFactory.create(name, api_key, model, **kwargs)
