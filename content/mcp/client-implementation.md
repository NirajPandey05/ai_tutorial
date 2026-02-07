# MCP Client Implementation

Learn how to build MCP clients that connect to and interact with MCP servers.

## Understanding MCP Clients

An MCP client connects to MCP servers to access their tools, resources, and prompts. Clients manage the connection lifecycle, send requests, and process responses.

```
┌─────────────────────────────────────────────────────────────────┐
│                    MCP Client Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     MCP CLIENT                           │    │
│  │                                                          │    │
│  │  ┌───────────┐  ┌──────────────┐  ┌─────────────────┐   │    │
│  │  │ Transport │  │   Session    │  │    Request      │   │    │
│  │  │  Manager  │  │   Handler    │  │    Builder      │   │    │
│  │  └───────────┘  └──────────────┘  └─────────────────┘   │    │
│  │        │               │                 │              │    │
│  │        ▼               ▼                 ▼              │    │
│  │  ┌─────────────────────────────────────────────────┐    │    │
│  │  │              Connection Manager                  │    │    │
│  │  │         (stdio / HTTP+SSE client)               │    │    │
│  │  └─────────────────────────────────────────────────┘    │    │
│  │                          │                              │    │
│  └──────────────────────────│──────────────────────────────┘    │
│                             │                                    │
│                             ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                      MCP SERVER                          │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Basic Client Setup

### Installing the MCP SDK

```bash
# Install the MCP Python SDK
pip install mcp

# Or with uv
uv add mcp
```

### Simple Client Example

```python
"""basic_client.py - A simple MCP client"""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # Configure server connection
    server_params = StdioServerParameters(
        command="python",           # Command to run
        args=["-m", "my_server"],   # Arguments
        env={"DEBUG": "true"}       # Environment variables
    )
    
    # Connect to server via stdio
    async with stdio_client(server_params) as (read_stream, write_stream):
        # Create session
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            result = await session.initialize()
            
            print(f"Connected to: {result.serverInfo.name}")
            print(f"Version: {result.serverInfo.version}")
            print(f"Capabilities: {result.capabilities}")
            
            # Now you can interact with the server
            # ...

if __name__ == "__main__":
    asyncio.run(main())
```

## Client Operations

### Listing Available Capabilities

```python
async def explore_server(session: ClientSession):
    """Discover what the server offers."""
    
    # List available tools
    print("=== TOOLS ===")
    tools_result = await session.list_tools()
    for tool in tools_result.tools:
        print(f"\n📧 {tool.name}")
        print(f"   Description: {tool.description}")
        print(f"   Schema: {tool.inputSchema}")
    
    # List available resources
    print("\n=== RESOURCES ===")
    resources_result = await session.list_resources()
    for resource in resources_result.resources:
        print(f"\n📁 {resource.uri}")
        print(f"   Name: {resource.name}")
        print(f"   Type: {resource.mimeType}")
    
    # List available prompts
    print("\n=== PROMPTS ===")
    prompts_result = await session.list_prompts()
    for prompt in prompts_result.prompts:
        print(f"\n💬 {prompt.name}")
        print(f"   Description: {prompt.description}")
        if prompt.arguments:
            print(f"   Arguments: {[a.name for a in prompt.arguments]}")
```

### Calling Tools

```python
async def call_tools_example(session: ClientSession):
    """Demonstrate tool calling."""
    
    # Call a tool with arguments
    result = await session.call_tool(
        name="get_weather",
        arguments={
            "location": "San Francisco",
            "units": "fahrenheit"
        }
    )
    
    # Process the response
    for content in result.content:
        if content.type == "text":
            print(f"Result: {content.text}")
        elif content.type == "image":
            print(f"Image: {content.mimeType}")
    
    # Check if the tool reported an error
    if result.isError:
        print(f"Tool error occurred")


async def call_tool_with_retry(
    session: ClientSession,
    name: str,
    arguments: dict,
    max_retries: int = 3
):
    """Call a tool with retry logic."""
    
    for attempt in range(max_retries):
        try:
            result = await session.call_tool(name, arguments)
            
            if not result.isError:
                return result
            
            # Tool returned an error, may retry
            print(f"Tool error on attempt {attempt + 1}")
            
        except Exception as e:
            print(f"Exception on attempt {attempt + 1}: {e}")
            
            if attempt == max_retries - 1:
                raise
        
        # Wait before retry
        await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

### Reading Resources

```python
async def read_resources_example(session: ClientSession):
    """Demonstrate resource reading."""
    
    # Read a specific resource
    result = await session.read_resource("file://config.json")
    
    for content in result.contents:
        if content.type == "text":
            print(f"Content ({content.mimeType}):")
            print(content.text)
        elif content.type == "blob":
            print(f"Binary content: {len(content.data)} bytes")


async def read_multiple_resources(
    session: ClientSession,
    uris: list[str]
) -> dict[str, str]:
    """Read multiple resources efficiently."""
    
    results = {}
    
    # Read in parallel using asyncio.gather
    tasks = [session.read_resource(uri) for uri in uris]
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    for uri, response in zip(uris, responses):
        if isinstance(response, Exception):
            results[uri] = f"Error: {response}"
        else:
            results[uri] = response.contents[0].text
    
    return results
```

### Using Prompts

```python
async def use_prompts_example(session: ClientSession):
    """Demonstrate prompt usage."""
    
    # Get a prompt with arguments
    result = await session.get_prompt(
        name="code_review",
        arguments={
            "file_path": "src/main.py",
            "focus": "security"
        }
    )
    
    print(f"Prompt: {result.description}")
    print("\nMessages to send to LLM:")
    
    for message in result.messages:
        print(f"\n[{message.role}]:")
        if hasattr(message.content, 'text'):
            print(message.content.text)
```

## Advanced Client Patterns

### Multi-Server Client

Connect to and manage multiple MCP servers:

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class ServerConnection:
    """Represents a connection to an MCP server."""
    name: str
    session: ClientSession
    capabilities: dict

class MultiServerClient:
    """Client that manages multiple MCP server connections."""
    
    def __init__(self):
        self._connections: dict[str, ServerConnection] = {}
    
    async def connect(
        self,
        name: str,
        command: str,
        args: list[str] = None,
        env: dict = None
    ) -> ServerConnection:
        """Connect to a server."""
        
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env or {}
        )
        
        # Create connection
        read, write = await stdio_client(params).__aenter__()
        session = await ClientSession(read, write).__aenter__()
        
        # Initialize
        result = await session.initialize()
        
        connection = ServerConnection(
            name=name,
            session=session,
            capabilities=result.capabilities
        )
        
        self._connections[name] = connection
        return connection
    
    async def disconnect(self, name: str):
        """Disconnect from a server."""
        if name in self._connections:
            conn = self._connections.pop(name)
            await conn.session.__aexit__(None, None, None)
    
    async def disconnect_all(self):
        """Disconnect from all servers."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)
    
    def get_connection(self, name: str) -> Optional[ServerConnection]:
        """Get a server connection by name."""
        return self._connections.get(name)
    
    async def call_tool_on_server(
        self,
        server: str,
        tool: str,
        arguments: dict
    ):
        """Call a tool on a specific server."""
        conn = self._connections.get(server)
        if not conn:
            raise ValueError(f"Not connected to server: {server}")
        
        return await conn.session.call_tool(tool, arguments)
    
    async def find_tool(self, tool_name: str) -> Optional[tuple[str, any]]:
        """Find which server has a specific tool."""
        for name, conn in self._connections.items():
            tools = await conn.session.list_tools()
            for tool in tools.tools:
                if tool.name == tool_name:
                    return (name, tool)
        return None


# Usage
async def multi_server_example():
    client = MultiServerClient()
    
    try:
        # Connect to multiple servers
        await client.connect(
            name="files",
            command="python",
            args=["-m", "file_server"]
        )
        
        await client.connect(
            name="weather",
            command="python",
            args=["-m", "weather_server"]
        )
        
        # Use tools from different servers
        weather = await client.call_tool_on_server(
            "weather", "get_weather", {"location": "NYC"}
        )
        
        files = await client.call_tool_on_server(
            "files", "list_files", {"path": "."}
        )
        
    finally:
        await client.disconnect_all()
```

### Client with Caching

```python
from datetime import datetime, timedelta
from typing import Any

class CachedMCPClient:
    """MCP client with response caching."""
    
    def __init__(self, session: ClientSession, cache_ttl: timedelta = None):
        self.session = session
        self.cache_ttl = cache_ttl or timedelta(minutes=5)
        self._cache: dict[str, tuple[datetime, Any]] = {}
    
    def _cache_key(self, operation: str, *args) -> str:
        """Generate cache key."""
        return f"{operation}:{':'.join(str(a) for a in args)}"
    
    def _is_cached(self, key: str) -> bool:
        """Check if a key is in cache and valid."""
        if key not in self._cache:
            return False
        cached_time, _ = self._cache[key]
        return datetime.now() - cached_time < self.cache_ttl
    
    def _get_cached(self, key: str) -> Any:
        """Get cached value."""
        return self._cache[key][1]
    
    def _set_cached(self, key: str, value: Any):
        """Set cached value."""
        self._cache[key] = (datetime.now(), value)
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries."""
        if pattern is None:
            self._cache.clear()
        else:
            keys_to_remove = [
                k for k in self._cache 
                if pattern in k
            ]
            for key in keys_to_remove:
                del self._cache[key]
    
    async def list_tools(self):
        """List tools with caching."""
        key = self._cache_key("list_tools")
        
        if self._is_cached(key):
            return self._get_cached(key)
        
        result = await self.session.list_tools()
        self._set_cached(key, result)
        return result
    
    async def list_resources(self):
        """List resources with caching."""
        key = self._cache_key("list_resources")
        
        if self._is_cached(key):
            return self._get_cached(key)
        
        result = await self.session.list_resources()
        self._set_cached(key, result)
        return result
    
    async def read_resource(self, uri: str, use_cache: bool = True):
        """Read resource with optional caching."""
        key = self._cache_key("read_resource", uri)
        
        if use_cache and self._is_cached(key):
            return self._get_cached(key)
        
        result = await self.session.read_resource(uri)
        
        if use_cache:
            self._set_cached(key, result)
        
        return result
    
    async def call_tool(self, name: str, arguments: dict):
        """Call tool (not cached - tools have side effects)."""
        # Invalidate related cache entries after tool calls
        result = await self.session.call_tool(name, arguments)
        
        # Invalidate resources cache since tools may modify data
        self.invalidate("read_resource")
        self.invalidate("list_resources")
        
        return result
```

### Event-Driven Client

```python
from typing import Callable, Any
from enum import Enum

class MCPEvent(Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    TOOL_CALLED = "tool_called"
    RESOURCE_READ = "resource_read"
    ERROR = "error"

class EventDrivenClient:
    """MCP client with event handling."""
    
    def __init__(self):
        self._handlers: dict[MCPEvent, list[Callable]] = {
            event: [] for event in MCPEvent
        }
        self._session: Optional[ClientSession] = None
    
    def on(self, event: MCPEvent, handler: Callable):
        """Register an event handler."""
        self._handlers[event].append(handler)
        return self  # Allow chaining
    
    async def _emit(self, event: MCPEvent, data: Any = None):
        """Emit an event to all handlers."""
        for handler in self._handlers[event]:
            if asyncio.iscoroutinefunction(handler):
                await handler(data)
            else:
                handler(data)
    
    async def connect(self, params: StdioServerParameters):
        """Connect to server."""
        try:
            read, write = await stdio_client(params).__aenter__()
            self._session = await ClientSession(read, write).__aenter__()
            
            result = await self._session.initialize()
            
            await self._emit(MCPEvent.CONNECTED, {
                "server": result.serverInfo.name,
                "version": result.serverInfo.version
            })
            
        except Exception as e:
            await self._emit(MCPEvent.ERROR, {"error": str(e)})
            raise
    
    async def call_tool(self, name: str, arguments: dict):
        """Call a tool with event emission."""
        try:
            result = await self._session.call_tool(name, arguments)
            
            await self._emit(MCPEvent.TOOL_CALLED, {
                "tool": name,
                "arguments": arguments,
                "success": not result.isError
            })
            
            return result
            
        except Exception as e:
            await self._emit(MCPEvent.ERROR, {
                "operation": "call_tool",
                "tool": name,
                "error": str(e)
            })
            raise


# Usage
async def event_driven_example():
    client = EventDrivenClient()
    
    # Register handlers
    client.on(MCPEvent.CONNECTED, lambda data: print(f"Connected: {data}"))
    client.on(MCPEvent.TOOL_CALLED, lambda data: print(f"Tool called: {data}"))
    client.on(MCPEvent.ERROR, lambda data: print(f"Error: {data}"))
    
    await client.connect(StdioServerParameters(
        command="python",
        args=["-m", "my_server"]
    ))
    
    result = await client.call_tool("hello", {"name": "World"})
```

## HTTP+SSE Client

For servers using HTTP transport:

```python
import httpx
import json
from typing import AsyncIterator

class HTTPMCPClient:
    """MCP client for HTTP+SSE transport."""
    
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = None
        self._session_id = None
    
    async def __aenter__(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0
        )
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    async def initialize(self) -> dict:
        """Initialize the session."""
        response = await self._client.post("/mcp/initialize", json={
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "http-client",
                "version": "1.0.0"
            }
        })
        response.raise_for_status()
        
        result = response.json()
        self._session_id = result.get("sessionId")
        return result
    
    async def list_tools(self) -> list:
        """List available tools."""
        response = await self._client.post("/mcp/tools/list", json={})
        response.raise_for_status()
        return response.json()["tools"]
    
    async def call_tool(self, name: str, arguments: dict) -> dict:
        """Call a tool."""
        response = await self._client.post("/mcp/tools/call", json={
            "name": name,
            "arguments": arguments
        })
        response.raise_for_status()
        return response.json()
    
    async def read_resource(self, uri: str) -> dict:
        """Read a resource."""
        response = await self._client.post("/mcp/resources/read", json={
            "uri": uri
        })
        response.raise_for_status()
        return response.json()
    
    async def subscribe_sse(self, uri: str) -> AsyncIterator[dict]:
        """Subscribe to SSE updates."""
        async with self._client.stream(
            "GET",
            f"/mcp/sse?uri={uri}",
            headers={"Accept": "text/event-stream"}
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = json.loads(line[6:])
                    yield data


# Usage
async def http_client_example():
    async with HTTPMCPClient("http://localhost:8000", api_key="secret") as client:
        await client.initialize()
        
        tools = await client.list_tools()
        print(f"Available tools: {[t['name'] for t in tools]}")
        
        result = await client.call_tool("hello", {"name": "World"})
        print(f"Result: {result}")
```

## Best Practices

```yaml
client_best_practices:
  connection:
    - "Always use async context managers"
    - "Handle connection failures gracefully"
    - "Implement reconnection logic"
    - "Clean up resources on exit"
  
  error_handling:
    - "Catch and handle specific exceptions"
    - "Implement retry with exponential backoff"
    - "Log errors for debugging"
    - "Provide meaningful error messages"
  
  performance:
    - "Cache tool and resource lists"
    - "Use parallel requests when possible"
    - "Implement request timeouts"
    - "Monitor and limit request rates"
  
  security:
    - "Validate server responses"
    - "Use secure transport when possible"
    - "Don't log sensitive arguments"
    - "Implement proper authentication"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│              MCP Client Implementation Summary                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Core Operations:                                                │
│    • Initialize session with server                             │
│    • List and call tools                                        │
│    • List and read resources                                    │
│    • Get and use prompts                                        │
│                                                                  │
│  Advanced Patterns:                                              │
│    • Multi-server client                                        │
│    • Response caching                                           │
│    • Event-driven architecture                                  │
│    • HTTP+SSE client                                            │
│                                                                  │
│  Key Considerations:                                             │
│    • Always clean up connections                                │
│    • Handle errors appropriately                                │
│    • Cache where beneficial                                     │
│    • Log for debugging                                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [Build an MCP Client Lab](/learn/mcp/mcp-clients/client-lab) →
