# MCP Architecture and Components

Understanding the building blocks of the Model Context Protocol.

## Architecture Overview

MCP follows a client-server architecture where **hosts** (AI applications) connect to **servers** (tool providers) through a standardized protocol.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MCP Architecture                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                           HOST                                   │    │
│  │                    (Claude Desktop, IDE)                         │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │                      MCP CLIENT                          │    │    │
│  │  │              (Protocol Implementation)                   │    │    │
│  │  └──────────────────────┬──────────────────────────────────┘    │    │
│  └─────────────────────────┼────────────────────────────────────────┘    │
│                            │                                             │
│                   Transport Layer                                        │
│              (stdio, HTTP+SSE, WebSocket)                               │
│                            │                                             │
│     ┌──────────────────────┼──────────────────────────┐                 │
│     │                      │                          │                  │
│     ▼                      ▼                          ▼                  │
│  ┌────────────┐     ┌────────────┐            ┌────────────┐            │
│  │ MCP Server │     │ MCP Server │            │ MCP Server │            │
│  │  (GitHub)  │     │   (Slack)  │            │ (Database) │            │
│  └────────────┘     └────────────┘            └────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Host

The **host** is the AI application that users interact with directly.

```yaml
host_responsibilities:
  - "Manage user interface and experience"
  - "Handle LLM API calls and responses"
  - "Create and manage MCP client connections"
  - "Enforce security policies and permissions"
  - "Coordinate between multiple MCP servers"

examples:
  - "Claude Desktop"
  - "VS Code with Copilot"
  - "Cursor IDE"
  - "Custom AI applications"
```

```python
# Host example: AI application managing MCP connections
class AIHost:
    def __init__(self):
        self.mcp_clients = {}  # Connected MCP servers
        self.llm_client = OpenAI()
    
    async def connect_server(self, name: str, command: str):
        """Connect to an MCP server."""
        client = MCPClient()
        await client.connect(command)
        self.mcp_clients[name] = client
        
    async def get_all_tools(self):
        """Aggregate tools from all connected servers."""
        all_tools = []
        for name, client in self.mcp_clients.items():
            tools = await client.list_tools()
            for tool in tools:
                tool.server = name  # Track which server owns this tool
                all_tools.append(tool)
        return all_tools
    
    async def chat(self, user_message: str):
        """Process user message with MCP tool access."""
        tools = await self.get_all_tools()
        
        # LLM decides if/which tools to use
        response = await self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": user_message}],
            tools=self._format_tools(tools)
        )
        
        # Execute tool calls if needed
        if response.tool_calls:
            for call in response.tool_calls:
                result = await self.execute_tool(call)
                # Continue conversation with result...
```

### 2. Client

The **client** is the protocol implementation within the host that communicates with MCP servers.

```yaml
client_responsibilities:
  - "Establish connections to MCP servers"
  - "Handle protocol handshake and capability negotiation"
  - "Send requests (list tools, call tool, etc.)"
  - "Receive and process responses"
  - "Manage connection lifecycle"
```

```python
from mcp import ClientSession, StdioServerParameters

class MCPClient:
    """MCP Client implementation."""
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.server_capabilities = None
    
    async def connect(self, command: str, args: list = None):
        """Connect to an MCP server via stdio."""
        server_params = StdioServerParameters(
            command=command,
            args=args or []
        )
        
        # Create session and initialize
        self.session = await ClientSession(server_params).__aenter__()
        
        # Capability negotiation
        init_result = await self.session.initialize()
        self.server_capabilities = init_result.capabilities
        
        print(f"Connected! Server capabilities: {self.server_capabilities}")
    
    async def list_tools(self):
        """Get available tools from server."""
        if not self.session:
            raise RuntimeError("Not connected")
        return await self.session.list_tools()
    
    async def call_tool(self, name: str, arguments: dict):
        """Execute a tool on the server."""
        return await self.session.call_tool(name, arguments)
    
    async def list_resources(self):
        """Get available resources from server."""
        return await self.session.list_resources()
    
    async def read_resource(self, uri: str):
        """Read a specific resource."""
        return await self.session.read_resource(uri)
```

### 3. Server

The **server** exposes tools, resources, and prompts to MCP clients.

```yaml
server_responsibilities:
  - "Declare available capabilities (tools, resources, prompts)"
  - "Execute tool calls and return results"
  - "Provide resource data when requested"
  - "Handle errors gracefully"
  - "Manage state if needed"
```

```python
from mcp.server import Server
from mcp.types import Tool, Resource, TextContent

# Create server instance
server = Server("example-server")

# Declare tools
@server.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        )
    ]

# Implement tools
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        city = arguments["city"]
        weather = fetch_weather(city)  # Your implementation
        return [TextContent(type="text", text=f"Weather in {city}: {weather}")]
    raise ValueError(f"Unknown tool: {name}")

# Declare resources
@server.list_resources()
async def list_resources():
    return [
        Resource(
            uri="weather://current/summary",
            name="Current Weather Summary",
            description="Summary of weather across monitored cities",
            mimeType="text/plain"
        )
    ]

# Implement resource reading
@server.read_resource()
async def read_resource(uri: str):
    if uri == "weather://current/summary":
        summary = generate_weather_summary()
        return [TextContent(type="text", text=summary)]
    raise ValueError(f"Unknown resource: {uri}")
```

## Protocol Layer: JSON-RPC 2.0

MCP uses JSON-RPC 2.0 for message formatting, providing a standard request-response pattern.

```yaml
json_rpc_benefits:
  - "Well-established standard"
  - "Simple request/response pattern"
  - "Built-in error handling"
  - "Language agnostic"
  - "Easy to debug and log"
```

### Request Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "city": "San Francisco"
    }
  }
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "Weather in San Francisco: 65°F, Partly Cloudy"
      }
    ]
  }
}
```

### Error Response

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": {
      "details": "Missing required parameter: city"
    }
  }
}
```

## Transport Layers

MCP supports multiple transport mechanisms for different deployment scenarios.

### 1. stdio (Standard I/O)

```yaml
stdio:
  description: "Communication via stdin/stdout"
  best_for: "Local servers, desktop apps, development"
  advantages:
    - "Simple to implement"
    - "No network configuration"
    - "Secure (local process)"
  limitations:
    - "Single client per server instance"
    - "Local only"
```

```python
# Client connecting via stdio
from mcp import StdioServerParameters

params = StdioServerParameters(
    command="python",
    args=["my_mcp_server.py"],
    env={"API_KEY": "secret"}  # Environment variables
)

# Server running via stdio
from mcp.server.stdio import stdio_server

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)
```

### 2. HTTP with SSE (Server-Sent Events)

```yaml
http_sse:
  description: "HTTP POST for requests, SSE for responses"
  best_for: "Remote servers, web deployments, multi-client"
  advantages:
    - "Works over network"
    - "Multiple clients supported"
    - "Web-friendly"
    - "Streaming responses"
  limitations:
    - "More complex setup"
    - "Requires HTTP server"
```

```python
# Server with HTTP+SSE transport
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route

transport = SseServerTransport("/messages")

async def handle_sse(request):
    async with transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(streams[0], streams[1])

app = Starlette(routes=[
    Route("/sse", endpoint=handle_sse),
    Route("/messages", endpoint=transport.handle_post_message, methods=["POST"])
])
```

### 3. WebSocket (Future)

```yaml
websocket:
  description: "Full-duplex WebSocket connection"
  best_for: "Real-time bidirectional communication"
  status: "Planned for future MCP versions"
  advantages:
    - "True bidirectional"
    - "Lower latency"
    - "Efficient for high-frequency communication"
```

## Capability Negotiation

When a client connects to a server, they negotiate capabilities to understand what features each supports.

```python
# Capability exchange during initialization
initialization_flow = """
Client                              Server
  │                                    │
  │─────── initialize request ────────►│
  │        {protocolVersion,           │
  │         capabilities,              │
  │         clientInfo}                │
  │                                    │
  │◄────── initialize response ────────│
  │        {protocolVersion,           │
  │         capabilities,              │
  │         serverInfo}                │
  │                                    │
  │─────── initialized notification ──►│
  │        (handshake complete)        │
  │                                    │
"""
```

```python
# Server capabilities declaration
server_capabilities = {
    "tools": {},           # Supports tools
    "resources": {
        "subscribe": True  # Supports resource subscriptions
    },
    "prompts": {},         # Supports prompts
    "logging": {}          # Supports logging
}

# Client capabilities declaration  
client_capabilities = {
    "roots": {
        "listChanged": True  # Can handle root changes
    },
    "sampling": {}  # Can handle sampling requests
}
```

## Message Types

MCP defines several message categories:

### Requests (Client → Server)

| Method | Description |
|--------|-------------|
| `initialize` | Start connection, exchange capabilities |
| `tools/list` | Get available tools |
| `tools/call` | Execute a tool |
| `resources/list` | Get available resources |
| `resources/read` | Read resource content |
| `prompts/list` | Get available prompts |
| `prompts/get` | Get a specific prompt |

### Notifications

```yaml
notifications:
  description: "One-way messages (no response expected)"
  
  client_to_server:
    - "initialized"  # Handshake complete
    - "cancelled"    # Cancel pending request
  
  server_to_client:
    - "tools/list_changed"     # Tools updated
    - "resources/list_changed" # Resources updated
    - "prompts/list_changed"   # Prompts updated
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Complete MCP Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  USER                                                                    │
│    │                                                                     │
│    ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                           HOST                                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │    │
│  │  │     UI       │  │   LLM API    │  │   Security   │          │    │
│  │  │  Interface   │  │   Client     │  │   Policy     │          │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │    │
│  │                                                                  │    │
│  │  ┌──────────────────────────────────────────────────────────┐   │    │
│  │  │                    MCP CLIENT LAYER                       │   │    │
│  │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐         │   │    │
│  │  │  │  Client 1  │  │  Client 2  │  │  Client 3  │         │   │    │
│  │  │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘         │   │    │
│  │  └────────┼───────────────┼───────────────┼─────────────────┘   │    │
│  └───────────┼───────────────┼───────────────┼──────────────────────┘    │
│              │               │               │                           │
│         ┌────┴────┐     ┌────┴────┐     ┌────┴────┐                     │
│         │  stdio  │     │  HTTP   │     │  stdio  │    TRANSPORTS       │
│         └────┬────┘     └────┬────┘     └────┬────┘                     │
│              │               │               │                           │
│  ┌───────────┴───────┐  ┌────┴────────┐  ┌──┴─────────────┐            │
│  │    MCP SERVER     │  │ MCP SERVER  │  │   MCP SERVER   │            │
│  │    (GitHub)       │  │  (Remote)   │  │  (Filesystem)  │            │
│  │                   │  │             │  │                │            │
│  │  Tools:           │  │  Tools:     │  │  Tools:        │            │
│  │  - create_issue   │  │  - query    │  │  - read_file   │            │
│  │  - list_repos     │  │  - insert   │  │  - write_file  │            │
│  │                   │  │             │  │  - search      │            │
│  │  Resources:       │  │  Resources: │  │                │            │
│  │  - repo://        │  │  - db://    │  │  Resources:    │            │
│  │                   │  │             │  │  - file://     │            │
│  └───────────────────┘  └─────────────┘  └────────────────┘            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Summary

```yaml
mcp_architecture_summary:
  components:
    host: "AI application (Claude, VS Code)"
    client: "Protocol implementation in host"
    server: "Tool/service provider"
    transport: "Communication channel (stdio, HTTP)"
  
  protocol:
    format: "JSON-RPC 2.0"
    messages: "Requests, responses, notifications"
    lifecycle: "Initialize → capabilities → operate → close"
  
  capabilities:
    tools: "Executable functions"
    resources: "Readable data"
    prompts: "Template prompts"
    
  key_insight: "Clean separation between AI apps and tools"
```

Next: [When to Use MCP](/learn/mcp/mcp-intro/when-to-use) →
