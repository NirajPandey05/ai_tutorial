# MCP Transport Layer

The transport layer defines how MCP clients and servers communicate. Understanding transports is key to deploying MCP in different environments.

## Transport Overview

MCP supports multiple transport mechanisms, each suited to different deployment scenarios:

```yaml
transport_types:
  stdio:
    description: "Communication via standard input/output"
    use_cases: "Local development, desktop apps, CLI tools"
    
  http_sse:
    description: "HTTP POST for requests, Server-Sent Events for responses"
    use_cases: "Web deployments, remote servers, cloud hosting"
    
  websocket:
    description: "Full-duplex WebSocket connection"
    status: "Planned for future MCP versions"
```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    MCP Transport Layer Architecture                      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        APPLICATION LAYER                         │    │
│  │              (Tools, Resources, Prompts, Messages)               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  │                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                        PROTOCOL LAYER                            │    │
│  │                       (JSON-RPC 2.0)                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                  │                                       │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                       TRANSPORT LAYER                            │    │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │    │
│  │  │     stdio     │  │   HTTP+SSE    │  │   WebSocket   │       │    │
│  │  │  (stdin/out)  │  │  (REST+Push)  │  │  (Planned)    │       │    │
│  │  └───────────────┘  └───────────────┘  └───────────────┘       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## stdio Transport

The **stdio** transport uses standard input/output streams for communication. It's the simplest and most common transport for local MCP servers.

### How stdio Works

```
┌────────────────────────────────────────────────────────────────┐
│                    stdio Transport Flow                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CLIENT (Host)                    SERVER (Subprocess)         │
│        │                                  │                    │
│        │──── spawn subprocess ───────────►│                    │
│        │                                  │                    │
│        │──── JSON-RPC via stdin ─────────►│                    │
│        │                                  │                    │
│        │◄─── JSON-RPC via stdout ─────────│                    │
│        │                                  │                    │
│        │◄─── Errors via stderr ───────────│                    │
│        │                                  │                    │
└────────────────────────────────────────────────────────────────┘
```

### Client-Side stdio Setup

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def connect_stdio_server():
    """Connect to an MCP server via stdio."""
    
    # Define server parameters
    server_params = StdioServerParameters(
        command="python",              # Command to run
        args=["my_mcp_server.py"],     # Command arguments
        env={                          # Environment variables
            "API_KEY": "secret",
            "DEBUG": "true"
        },
        cwd="/path/to/server"          # Working directory (optional)
    )
    
    # Create and connect
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            
            # Now use the session
            tools = await session.list_tools()
            print(f"Connected! Tools: {[t.name for t in tools.tools]}")
            
            return session
```

### Server-Side stdio Setup

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server

# Create your server
server = Server("my-server")

# ... define tools, resources, prompts ...

async def main():
    """Run the MCP server with stdio transport."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### stdio Configuration (Claude Desktop Example)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"],
      "env": {}
    },
    "my-custom-server": {
      "command": "python",
      "args": ["/path/to/my_server.py"],
      "env": {
        "DATABASE_URL": "postgresql://..."
      }
    }
  }
}
```

### stdio Advantages and Limitations

```yaml
stdio_advantages:
  - "Simple to implement and debug"
  - "No network configuration needed"
  - "Secure (process isolation)"
  - "Works offline"
  - "Low latency for local operations"

stdio_limitations:
  - "Single client per server instance"
  - "Must spawn new process per connection"
  - "Local only (no remote access)"
  - "Process lifecycle management needed"
```

## HTTP with SSE Transport

The **HTTP+SSE** transport enables remote MCP servers over HTTP, using Server-Sent Events for streaming responses.

### How HTTP+SSE Works

```
┌────────────────────────────────────────────────────────────────┐
│                   HTTP+SSE Transport Flow                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   CLIENT                              SERVER                   │
│      │                                   │                     │
│      │──── GET /sse ────────────────────►│  (SSE connection)  │
│      │◄──── SSE stream opened ───────────│                     │
│      │                                   │                     │
│      │──── POST /messages ──────────────►│  (JSON-RPC req)    │
│      │                                   │                     │
│      │◄──── SSE event ───────────────────│  (JSON-RPC resp)   │
│      │                                   │                     │
│      │◄──── SSE notification ────────────│  (Server push)     │
│      │                                   │                     │
└────────────────────────────────────────────────────────────────┘
```

### Server-Side HTTP+SSE Setup

```python
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.responses import Response
import uvicorn

# Create MCP server
server = Server("http-server")

# ... define tools, resources, prompts ...

# Create SSE transport
sse_transport = SseServerTransport("/messages")

async def handle_sse(request):
    """Handle SSE connection requests."""
    async with sse_transport.connect_sse(
        request.scope,
        request.receive,
        request._send
    ) as streams:
        await server.run(
            streams[0],
            streams[1],
            server.create_initialization_options()
        )

async def handle_messages(request):
    """Handle JSON-RPC messages via POST."""
    return await sse_transport.handle_post_message(request.scope, request.receive, request._send)

# Create Starlette app
app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Client-Side HTTP+SSE Setup

```python
from mcp import ClientSession
from mcp.client.sse import sse_client

async def connect_http_server():
    """Connect to an MCP server via HTTP+SSE."""
    
    server_url = "http://localhost:8000/sse"
    
    async with sse_client(server_url) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            
            tools = await session.list_tools()
            print(f"Connected via HTTP! Tools: {[t.name for t in tools.tools]}")
            
            return session
```

### HTTP+SSE with Authentication

```python
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.authentication import AuthCredentials, SimpleUser

class APIKeyAuthBackend:
    async def authenticate(self, request):
        api_key = request.headers.get("Authorization", "").replace("Bearer ", "")
        
        if api_key == "valid-api-key":
            return AuthCredentials(["authenticated"]), SimpleUser("user")
        return None

# Add authentication middleware
app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Route("/messages", endpoint=handle_messages, methods=["POST"]),
    ],
    middleware=[
        Middleware(AuthenticationMiddleware, backend=APIKeyAuthBackend())
    ]
)

# Client with authentication
async def connect_with_auth():
    headers = {"Authorization": "Bearer valid-api-key"}
    
    async with sse_client(
        "http://localhost:8000/sse",
        headers=headers
    ) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            # ...
```

### HTTP+SSE Advantages and Limitations

```yaml
http_sse_advantages:
  - "Works over network"
  - "Multiple clients supported"
  - "Web-compatible"
  - "Can use existing HTTP infrastructure"
  - "Supports authentication/authorization"
  - "Scalable with load balancers"

http_sse_limitations:
  - "More complex setup"
  - "Higher latency than stdio"
  - "Requires proper error handling for network issues"
  - "SSE is one-way (server to client)"
```

## Message Framing

Both transports use newline-delimited JSON for message framing:

```python
# Message format (newline-delimited JSON)
message_format = """
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}\n
{"jsonrpc":"2.0","id":2,"method":"tools/list"}\n
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{...}}\n
"""

# Each message is a complete JSON object followed by newline
# This allows for simple parsing and streaming
```

### Protocol Messages

```python
# Request message
request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
        "name": "search_files",
        "arguments": {"pattern": "*.py"}
    }
}

# Response message
response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "content": [{"type": "text", "text": "Found 5 files..."}]
    }
}

# Notification (no id, no response expected)
notification = {
    "jsonrpc": "2.0",
    "method": "notifications/resources/list_changed"
}

# Error response
error = {
    "jsonrpc": "2.0",
    "id": 1,
    "error": {
        "code": -32602,
        "message": "Invalid params",
        "data": {"details": "Missing required field: name"}
    }
}
```

## Choosing a Transport

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Transport Selection Guide                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Use stdio when:                                                        │
│    ✓ Building desktop applications (Claude Desktop, IDEs)              │
│    ✓ Local development and testing                                      │
│    ✓ CLI tools that spawn MCP servers                                  │
│    ✓ Single-user scenarios                                              │
│    ✓ Maximum security (no network exposure)                            │
│                                                                          │
│  Use HTTP+SSE when:                                                     │
│    ✓ Deploying to cloud/remote servers                                 │
│    ✓ Multiple clients need to connect                                  │
│    ✓ Web application integration                                        │
│    ✓ Need authentication/authorization                                 │
│    ✓ Scaling across multiple instances                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Decision Matrix

| Factor | stdio | HTTP+SSE |
|--------|-------|----------|
| **Setup Complexity** | ⭐ Low | ⭐⭐⭐ Medium-High |
| **Network Support** | ❌ Local only | ✅ Full |
| **Multi-client** | ❌ One per process | ✅ Many |
| **Security** | ✅ Process isolation | ⚠️ Requires config |
| **Latency** | ⭐⭐⭐ Lowest | ⭐⭐ Low |
| **Debugging** | ⭐⭐⭐ Easy | ⭐⭐ Medium |
| **Production Ready** | ⭐⭐ Good | ⭐⭐⭐ Best |

## Implementing Custom Transports

If you need a custom transport (e.g., WebSocket, Unix sockets), you can implement the transport interface:

```python
from typing import AsyncIterator, Protocol
from mcp.types import JSONRPCMessage

class Transport(Protocol):
    """Interface for MCP transports."""
    
    async def read(self) -> AsyncIterator[JSONRPCMessage]:
        """Read messages from the transport."""
        ...
    
    async def write(self, message: JSONRPCMessage) -> None:
        """Write a message to the transport."""
        ...
    
    async def close(self) -> None:
        """Close the transport."""
        ...


class WebSocketTransport:
    """Example WebSocket transport implementation."""
    
    def __init__(self, websocket):
        self.websocket = websocket
    
    async def read(self) -> AsyncIterator[JSONRPCMessage]:
        async for message in self.websocket:
            yield json.loads(message)
    
    async def write(self, message: JSONRPCMessage) -> None:
        await self.websocket.send(json.dumps(message))
    
    async def close(self) -> None:
        await self.websocket.close()
```

## Error Handling

Both transports should handle errors gracefully:

```python
import asyncio
from mcp.shared.exceptions import McpError

async def robust_connection(params):
    """Connection with error handling and retry."""
    
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return session
                    
        except ConnectionError as e:
            if attempt < max_retries - 1:
                print(f"Connection failed, retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise
                
        except McpError as e:
            print(f"MCP protocol error: {e}")
            raise
            
        except Exception as e:
            print(f"Unexpected error: {e}")
            raise
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                  MCP Transport Layer Summary                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Transports = How clients and servers communicate               │
│                                                                  │
│  Available Transports:                                          │
│    • stdio: stdin/stdout for local processes                    │
│    • HTTP+SSE: HTTP POST + Server-Sent Events for remote        │
│    • WebSocket: Planned for future versions                     │
│                                                                  │
│  Message Format:                                                 │
│    • JSON-RPC 2.0                                               │
│    • Newline-delimited                                          │
│    • Request/response/notification patterns                     │
│                                                                  │
│  Selection Guide:                                                │
│    • stdio: Local, desktop, development                         │
│    • HTTP+SSE: Remote, cloud, multi-client                      │
│                                                                  │
│  Key Considerations:                                             │
│    • Security requirements                                       │
│    • Scalability needs                                          │
│    • Network constraints                                         │
│    • Client count                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [MCP Server Structure](/learn/mcp/mcp-servers/server-structure) →
