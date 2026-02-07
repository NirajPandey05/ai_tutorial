# MCP Server Structure

Learn how to structure an MCP server from the ground up, understanding the key components and patterns.

## Server Overview

An MCP server is a program that exposes tools, resources, and prompts to MCP clients. The server handles incoming requests, executes operations, and returns results.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MCP Server Architecture                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        MCP SERVER                                │    │
│  │                                                                  │    │
│  │  ┌────────────────────────────────────────────────────────┐     │    │
│  │  │                    Server Instance                      │     │    │
│  │  │                  (Name, Version, Info)                  │     │    │
│  │  └────────────────────────────────────────────────────────┘     │    │
│  │                              │                                   │    │
│  │  ┌───────────┬───────────────┼───────────────┬───────────┐      │    │
│  │  │           │               │               │           │      │    │
│  │  ▼           ▼               ▼               ▼           ▼      │    │
│  │  ┌─────┐  ┌─────────┐  ┌──────────┐  ┌─────────┐  ┌─────┐     │    │
│  │  │Tools│  │Resources│  │ Prompts  │  │Lifecycle│  │State│     │    │
│  │  └─────┘  └─────────┘  └──────────┘  └─────────┘  └─────┘     │    │
│  │                                                                  │    │
│  │  ┌────────────────────────────────────────────────────────┐     │    │
│  │  │                   Transport Layer                       │     │    │
│  │  │              (stdio or HTTP+SSE)                        │     │    │
│  │  └────────────────────────────────────────────────────────┘     │    │
│  │                                                                  │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Basic Server Template

Here's the minimal structure for an MCP server:

```python
"""
my_mcp_server.py - A basic MCP server template
"""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, Resource, Prompt, TextContent

# Create server instance
server = Server(
    name="my-server",           # Unique server name
    version="1.0.0"             # Server version
)

# Define capabilities via decorators
@server.list_tools()
async def list_tools():
    """Declare available tools."""
    return []

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Handle tool execution."""
    raise ValueError(f"Unknown tool: {name}")

@server.list_resources()
async def list_resources():
    """Declare available resources."""
    return []

@server.read_resource()
async def read_resource(uri: str):
    """Handle resource reading."""
    raise ValueError(f"Unknown resource: {uri}")

@server.list_prompts()
async def list_prompts():
    """Declare available prompts."""
    return []

@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    """Handle prompt requests."""
    raise ValueError(f"Unknown prompt: {name}")

# Entry point
async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

For larger servers, organize code into modules:

```
my_mcp_server/
├── pyproject.toml          # Project configuration
├── README.md               # Documentation
├── src/
│   └── my_mcp_server/
│       ├── __init__.py
│       ├── __main__.py     # Entry point
│       ├── server.py       # Server setup
│       ├── tools/          # Tool implementations
│       │   ├── __init__.py
│       │   ├── registry.py
│       │   ├── file_tools.py
│       │   └── api_tools.py
│       ├── resources/      # Resource implementations
│       │   ├── __init__.py
│       │   └── file_resources.py
│       ├── prompts/        # Prompt implementations
│       │   ├── __init__.py
│       │   └── templates.py
│       └── config.py       # Configuration
└── tests/
    └── test_server.py
```

### pyproject.toml

```toml
[project]
name = "my-mcp-server"
version = "1.0.0"
description = "My MCP Server"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
]

[project.scripts]
my-mcp-server = "my_mcp_server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### __main__.py

```python
"""Entry point for running as a module: python -m my_mcp_server"""

import asyncio
from .server import main

if __name__ == "__main__":
    asyncio.run(main())
```

### server.py

```python
"""MCP Server setup and configuration."""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .tools import tool_registry
from .resources import resource_registry
from .prompts import prompt_registry
from .config import settings

# Create server
server = Server(
    name=settings.server_name,
    version=settings.version
)

# Register handlers
@server.list_tools()
async def list_tools():
    return tool_registry.list_tools()

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    return await tool_registry.call_tool(name, arguments)

@server.list_resources()
async def list_resources():
    return resource_registry.list_resources()

@server.read_resource()
async def read_resource(uri: str):
    return await resource_registry.read_resource(uri)

@server.list_prompts()
async def list_prompts():
    return prompt_registry.list_prompts()

@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    return prompt_registry.get_prompt(name, arguments)

async def main():
    """Run the server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
```

## Server Configuration

### Configuration Module

```python
"""config.py - Server configuration"""

import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Settings:
    """Server configuration settings."""
    
    # Server info
    server_name: str = "my-mcp-server"
    version: str = "1.0.0"
    
    # Feature flags
    enable_tools: bool = True
    enable_resources: bool = True
    enable_prompts: bool = True
    
    # Security
    allowed_directories: list[str] = field(default_factory=lambda: [os.getcwd()])
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    
    # External services
    api_key: Optional[str] = None
    database_url: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            server_name=os.getenv("MCP_SERVER_NAME", "my-mcp-server"),
            version=os.getenv("MCP_VERSION", "1.0.0"),
            api_key=os.getenv("API_KEY"),
            database_url=os.getenv("DATABASE_URL"),
            allowed_directories=os.getenv("ALLOWED_DIRS", os.getcwd()).split(":"),
        )

# Global settings instance
settings = Settings.from_env()
```

### Using Configuration

```python
from .config import settings

@server.list_tools()
async def list_tools():
    tools = []
    
    if settings.enable_tools:
        tools.extend(tool_registry.list_tools())
    
    return tools

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    # Validate against security settings
    if "path" in arguments:
        path = arguments["path"]
        if not any(path.startswith(d) for d in settings.allowed_directories):
            raise ValueError("Path not in allowed directories")
    
    return await tool_registry.call_tool(name, arguments)
```

## Handler Registration Pattern

Use a registry pattern for managing handlers:

```python
"""tools/registry.py - Tool registry pattern"""

from typing import Callable, Any
from dataclasses import dataclass
from mcp.types import Tool, TextContent

@dataclass
class RegisteredTool:
    """A registered tool with its handler."""
    definition: Tool
    handler: Callable

class ToolRegistry:
    """Registry for managing tools."""
    
    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}
    
    def tool(
        self,
        name: str,
        description: str,
        input_schema: dict
    ):
        """Decorator to register a tool."""
        def decorator(func: Callable):
            self._tools[name] = RegisteredTool(
                definition=Tool(
                    name=name,
                    description=description,
                    inputSchema=input_schema
                ),
                handler=func
            )
            return func
        return decorator
    
    def list_tools(self) -> list[Tool]:
        """Get all registered tools."""
        return [t.definition for t in self._tools.values()]
    
    async def call_tool(self, name: str, arguments: dict) -> list[TextContent]:
        """Execute a tool by name."""
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        
        tool = self._tools[name]
        result = await tool.handler(arguments)
        
        # Ensure result is properly formatted
        if isinstance(result, str):
            return [TextContent(type="text", text=result)]
        elif isinstance(result, list):
            return result
        else:
            return [TextContent(type="text", text=str(result))]

# Global registry instance
tool_registry = ToolRegistry()

# Example tool registration
@tool_registry.tool(
    name="hello",
    description="Say hello to someone",
    input_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Name to greet"}
        },
        "required": ["name"]
    }
)
async def hello_tool(arguments: dict) -> str:
    name = arguments["name"]
    return f"Hello, {name}!"
```

## Lifecycle Hooks

Handle server lifecycle events:

```python
from contextlib import asynccontextmanager

class ManagedServer:
    """Server with lifecycle management."""
    
    def __init__(self):
        self.server = Server("managed-server")
        self._db_connection = None
        self._cache = {}
        
        # Register handlers
        self._register_handlers()
    
    async def startup(self):
        """Initialize resources on startup."""
        print("Server starting up...")
        
        # Connect to database
        self._db_connection = await create_db_connection()
        
        # Warm up cache
        self._cache = await load_initial_cache()
        
        print("Server ready!")
    
    async def shutdown(self):
        """Cleanup on shutdown."""
        print("Server shutting down...")
        
        # Close database connection
        if self._db_connection:
            await self._db_connection.close()
        
        # Clear cache
        self._cache.clear()
        
        print("Server stopped.")
    
    def _register_handlers(self):
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(name="query_db", description="Query the database", inputSchema={...})
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "query_db":
                # Use the managed connection
                result = await self._db_connection.execute(arguments["query"])
                return [TextContent(type="text", text=str(result))]
            raise ValueError(f"Unknown tool: {name}")
    
    @asynccontextmanager
    async def run_context(self):
        """Context manager for server lifecycle."""
        try:
            await self.startup()
            yield self.server
        finally:
            await self.shutdown()

async def main():
    managed = ManagedServer()
    
    async with managed.run_context() as server:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options()
            )
```

## Error Handling

Implement robust error handling:

```python
import logging
import traceback
from mcp.types import TextContent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerError(Exception):
    """Base class for server errors."""
    pass

class ToolExecutionError(ServerError):
    """Error during tool execution."""
    def __init__(self, tool_name: str, message: str, details: dict = None):
        self.tool_name = tool_name
        self.details = details or {}
        super().__init__(f"Tool '{tool_name}' failed: {message}")

class ResourceNotFoundError(ServerError):
    """Resource not found."""
    def __init__(self, uri: str):
        self.uri = uri
        super().__init__(f"Resource not found: {uri}")

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute tool with error handling."""
    logger.info(f"Calling tool: {name} with args: {arguments}")
    
    try:
        result = await tool_registry.call_tool(name, arguments)
        logger.info(f"Tool {name} completed successfully")
        return result
        
    except ValueError as e:
        # Known validation error
        logger.warning(f"Validation error in tool {name}: {e}")
        return [TextContent(
            type="text",
            text=f"Invalid input: {str(e)}"
        )]
        
    except ToolExecutionError as e:
        # Known tool error
        logger.error(f"Tool execution error: {e}")
        return [TextContent(
            type="text",
            text=f"Tool error: {str(e)}"
        )]
        
    except Exception as e:
        # Unexpected error
        logger.exception(f"Unexpected error in tool {name}")
        return [TextContent(
            type="text",
            text=f"An unexpected error occurred. Please try again."
        )]
```

## Testing Your Server

```python
"""tests/test_server.py - Server tests"""

import pytest
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

@pytest.fixture
async def mcp_session():
    """Create a test session with the server."""
    params = StdioServerParameters(
        command="python",
        args=["-m", "my_mcp_server"]
    )
    
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session

@pytest.mark.asyncio
async def test_list_tools(mcp_session):
    """Test that tools are listed correctly."""
    result = await mcp_session.list_tools()
    
    assert len(result.tools) > 0
    tool_names = [t.name for t in result.tools]
    assert "hello" in tool_names

@pytest.mark.asyncio
async def test_call_tool(mcp_session):
    """Test tool execution."""
    result = await mcp_session.call_tool(
        name="hello",
        arguments={"name": "World"}
    )
    
    assert len(result.content) == 1
    assert "Hello, World!" in result.content[0].text

@pytest.mark.asyncio
async def test_invalid_tool(mcp_session):
    """Test error handling for invalid tool."""
    with pytest.raises(Exception):
        await mcp_session.call_tool(
            name="nonexistent_tool",
            arguments={}
        )
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                   MCP Server Structure Summary                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Core Components:                                                │
│    • Server instance with name and version                      │
│    • Handler decorators for tools, resources, prompts           │
│    • Transport layer (stdio or HTTP)                            │
│    • Configuration management                                   │
│                                                                  │
│  Key Patterns:                                                   │
│    • Registry pattern for handlers                              │
│    • Lifecycle hooks for setup/teardown                         │
│    • Centralized error handling                                 │
│    • Modular project structure                                  │
│                                                                  │
│  Best Practices:                                                 │
│    • Use environment variables for config                       │
│    • Implement proper logging                                   │
│    • Handle errors gracefully                                   │
│    • Write tests for your server                                │
│    • Document your tools and resources                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [Implementing Tools](/learn/mcp/mcp-servers/implementing-tools) →
