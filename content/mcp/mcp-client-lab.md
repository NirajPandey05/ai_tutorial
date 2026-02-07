# Lab: Build an MCP Client

Create an interactive MCP client that can connect to servers and provide a REPL interface.

## Lab Overview

```yaml
objectives:
  - "Build a complete MCP client from scratch"
  - "Implement interactive REPL interface"
  - "Handle multiple server connections"
  - "Practice error handling and user feedback"

difficulty: Advanced
time: 30 minutes
prerequisites:
  - "MCP client basics"
  - "Python async programming"
  - "Terminal I/O"
```

## What We're Building

An **Interactive MCP Client** that provides:
- Connect to multiple MCP servers
- Interactive command interface
- Tool execution with argument parsing
- Resource browsing
- Prompt usage

```
┌─────────────────────────────────────────────────────────────────┐
│                  Interactive MCP Client                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  COMMANDS:                                                       │
│    connect <name> <command> [args]  - Connect to server         │
│    disconnect <name>                - Disconnect from server    │
│    servers                          - List connected servers    │
│    tools [server]                   - List available tools      │
│    call <tool> [args]               - Call a tool               │
│    resources [server]               - List resources            │
│    read <uri>                       - Read a resource           │
│    prompts [server]                 - List prompts              │
│    prompt <name> [args]             - Get a prompt              │
│    help                             - Show help                 │
│    exit                             - Exit client               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Part 1: Project Structure

```
mcp_client/
├── pyproject.toml
└── src/
    └── mcp_client/
        ├── __init__.py
        ├── __main__.py
        ├── client.py
        ├── repl.py
        └── commands.py
```

### pyproject.toml

```toml
[project]
name = "mcp-client"
version = "1.0.0"
description = "Interactive MCP Client"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
    "rich>=13.0.0",  # For beautiful terminal output
]

[project.scripts]
mcp-client = "mcp_client:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Part 2: The Client Core

```python
# src/mcp_client/client.py
"""Core MCP client implementation."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, Any
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@dataclass
class ServerConnection:
    """Represents an active server connection."""
    name: str
    command: str
    args: list[str]
    session: ClientSession
    read_stream: Any
    write_stream: Any
    server_info: dict = field(default_factory=dict)
    capabilities: dict = field(default_factory=dict)


class MCPClient:
    """Multi-server MCP client."""
    
    def __init__(self):
        self._connections: dict[str, ServerConnection] = {}
        self._active_server: Optional[str] = None
    
    @property
    def active_server(self) -> Optional[str]:
        """Get the currently active server name."""
        return self._active_server
    
    @active_server.setter
    def active_server(self, name: str):
        """Set the active server."""
        if name and name not in self._connections:
            raise ValueError(f"Not connected to server: {name}")
        self._active_server = name
    
    @property
    def servers(self) -> list[str]:
        """List all connected servers."""
        return list(self._connections.keys())
    
    def get_connection(self, name: str = None) -> Optional[ServerConnection]:
        """Get a server connection."""
        target = name or self._active_server
        if not target:
            return None
        return self._connections.get(target)
    
    async def connect(
        self,
        name: str,
        command: str,
        args: list[str] = None,
        env: dict = None
    ) -> ServerConnection:
        """Connect to an MCP server."""
        
        if name in self._connections:
            raise ValueError(f"Already connected to server: {name}")
        
        # Create server parameters
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env or {}
        )
        
        # Establish connection
        read_stream, write_stream = await stdio_client(params).__aenter__()
        session = await ClientSession(read_stream, write_stream).__aenter__()
        
        # Initialize
        init_result = await session.initialize()
        
        # Create connection object
        connection = ServerConnection(
            name=name,
            command=command,
            args=args or [],
            session=session,
            read_stream=read_stream,
            write_stream=write_stream,
            server_info={
                "name": init_result.serverInfo.name,
                "version": init_result.serverInfo.version
            },
            capabilities=init_result.capabilities or {}
        )
        
        self._connections[name] = connection
        
        # Set as active if first connection
        if not self._active_server:
            self._active_server = name
        
        return connection
    
    async def disconnect(self, name: str):
        """Disconnect from a server."""
        if name not in self._connections:
            raise ValueError(f"Not connected to server: {name}")
        
        conn = self._connections.pop(name)
        
        # Clean up session
        try:
            await conn.session.__aexit__(None, None, None)
        except Exception:
            pass
        
        # Update active server
        if self._active_server == name:
            self._active_server = next(iter(self._connections.keys()), None)
    
    async def disconnect_all(self):
        """Disconnect from all servers."""
        for name in list(self._connections.keys()):
            await self.disconnect(name)
    
    async def list_tools(self, server: str = None) -> list:
        """List tools from a server."""
        conn = self.get_connection(server)
        if not conn:
            raise ValueError("No active server connection")
        
        result = await conn.session.list_tools()
        return result.tools
    
    async def call_tool(
        self,
        name: str,
        arguments: dict,
        server: str = None
    ) -> Any:
        """Call a tool on a server."""
        conn = self.get_connection(server)
        if not conn:
            raise ValueError("No active server connection")
        
        return await conn.session.call_tool(name, arguments)
    
    async def list_resources(self, server: str = None) -> list:
        """List resources from a server."""
        conn = self.get_connection(server)
        if not conn:
            raise ValueError("No active server connection")
        
        result = await conn.session.list_resources()
        return result.resources
    
    async def read_resource(self, uri: str, server: str = None) -> Any:
        """Read a resource from a server."""
        conn = self.get_connection(server)
        if not conn:
            raise ValueError("No active server connection")
        
        return await conn.session.read_resource(uri)
    
    async def list_prompts(self, server: str = None) -> list:
        """List prompts from a server."""
        conn = self.get_connection(server)
        if not conn:
            raise ValueError("No active server connection")
        
        result = await conn.session.list_prompts()
        return result.prompts
    
    async def get_prompt(
        self,
        name: str,
        arguments: dict = None,
        server: str = None
    ) -> Any:
        """Get a prompt from a server."""
        conn = self.get_connection(server)
        if not conn:
            raise ValueError("No active server connection")
        
        return await conn.session.get_prompt(name, arguments)
```

## Part 3: Command Handlers

```python
# src/mcp_client/commands.py
"""Command handlers for the REPL."""

import json
import shlex
from typing import Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax

from .client import MCPClient

console = Console()


async def cmd_connect(client: MCPClient, args: list[str]) -> str:
    """Connect to an MCP server."""
    if len(args) < 2:
        return "Usage: connect <name> <command> [args...]"
    
    name = args[0]
    command = args[1]
    cmd_args = args[2:] if len(args) > 2 else []
    
    try:
        conn = await client.connect(name, command, cmd_args)
        return f"✅ Connected to {conn.server_info['name']} v{conn.server_info['version']}"
    except Exception as e:
        return f"❌ Failed to connect: {e}"


async def cmd_disconnect(client: MCPClient, args: list[str]) -> str:
    """Disconnect from a server."""
    if not args:
        if not client.active_server:
            return "No active server to disconnect"
        name = client.active_server
    else:
        name = args[0]
    
    try:
        await client.disconnect(name)
        return f"✅ Disconnected from {name}"
    except Exception as e:
        return f"❌ Failed to disconnect: {e}"


async def cmd_servers(client: MCPClient, args: list[str]) -> str:
    """List connected servers."""
    if not client.servers:
        return "No servers connected"
    
    table = Table(title="Connected Servers")
    table.add_column("Name", style="cyan")
    table.add_column("Server", style="green")
    table.add_column("Version", style="yellow")
    table.add_column("Active", style="magenta")
    
    for name in client.servers:
        conn = client.get_connection(name)
        is_active = "✓" if name == client.active_server else ""
        table.add_row(
            name,
            conn.server_info.get("name", "unknown"),
            conn.server_info.get("version", "?"),
            is_active
        )
    
    console.print(table)
    return ""


async def cmd_use(client: MCPClient, args: list[str]) -> str:
    """Set the active server."""
    if not args:
        if client.active_server:
            return f"Active server: {client.active_server}"
        return "No active server"
    
    try:
        client.active_server = args[0]
        return f"✅ Now using server: {args[0]}"
    except ValueError as e:
        return f"❌ {e}"


async def cmd_tools(client: MCPClient, args: list[str]) -> str:
    """List available tools."""
    server = args[0] if args else None
    
    try:
        tools = await client.list_tools(server)
    except Exception as e:
        return f"❌ Failed to list tools: {e}"
    
    if not tools:
        return "No tools available"
    
    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white", max_width=60)
    
    for tool in tools:
        desc = tool.description or ""
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(tool.name, desc.split("\n")[0])
    
    console.print(table)
    return f"\n{len(tools)} tool(s) available. Use 'tool <name>' for details."


async def cmd_tool_info(client: MCPClient, args: list[str]) -> str:
    """Show detailed tool information."""
    if not args:
        return "Usage: tool <name>"
    
    tool_name = args[0]
    
    try:
        tools = await client.list_tools()
        tool = next((t for t in tools if t.name == tool_name), None)
        
        if not tool:
            return f"❌ Tool not found: {tool_name}"
        
        panel = Panel(
            f"[bold]{tool.name}[/bold]\n\n"
            f"{tool.description or 'No description'}\n\n"
            f"[dim]Input Schema:[/dim]\n"
            f"{json.dumps(tool.inputSchema, indent=2)}",
            title="Tool Details"
        )
        console.print(panel)
        return ""
        
    except Exception as e:
        return f"❌ Failed to get tool info: {e}"


async def cmd_call(client: MCPClient, args: list[str]) -> str:
    """Call a tool."""
    if not args:
        return "Usage: call <tool_name> [json_arguments]"
    
    tool_name = args[0]
    
    # Parse arguments
    if len(args) > 1:
        try:
            arguments = json.loads(" ".join(args[1:]))
        except json.JSONDecodeError:
            # Try parsing as key=value pairs
            arguments = {}
            for arg in args[1:]:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    # Try to parse value as JSON, fall back to string
                    try:
                        arguments[key] = json.loads(value)
                    except json.JSONDecodeError:
                        arguments[key] = value
                else:
                    return f"❌ Invalid argument format: {arg} (use key=value)"
    else:
        arguments = {}
    
    try:
        result = await client.call_tool(tool_name, arguments)
        
        output = f"[bold green]Tool: {tool_name}[/bold green]\n"
        
        for content in result.content:
            if content.type == "text":
                output += f"\n{content.text}"
            elif content.type == "image":
                output += f"\n[Image: {content.mimeType}]"
        
        if result.isError:
            output = f"[red]Tool Error:[/red]\n{output}"
        
        console.print(Panel(output, title="Result"))
        return ""
        
    except Exception as e:
        return f"❌ Failed to call tool: {e}"


async def cmd_resources(client: MCPClient, args: list[str]) -> str:
    """List available resources."""
    server = args[0] if args else None
    
    try:
        resources = await client.list_resources(server)
    except Exception as e:
        return f"❌ Failed to list resources: {e}"
    
    if not resources:
        return "No resources available"
    
    table = Table(title="Available Resources")
    table.add_column("URI", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Type", style="yellow")
    
    for resource in resources:
        table.add_row(
            resource.uri,
            resource.name,
            resource.mimeType or "unknown"
        )
    
    console.print(table)
    return f"\n{len(resources)} resource(s) available"


async def cmd_read(client: MCPClient, args: list[str]) -> str:
    """Read a resource."""
    if not args:
        return "Usage: read <uri>"
    
    uri = args[0]
    
    try:
        result = await client.read_resource(uri)
        
        for content in result.contents:
            if content.type == "text":
                # Try to syntax highlight if it looks like code
                if any(uri.endswith(ext) for ext in [".py", ".js", ".json", ".yaml"]):
                    ext = uri.split(".")[-1]
                    lang_map = {"py": "python", "js": "javascript"}
                    lang = lang_map.get(ext, ext)
                    syntax = Syntax(content.text, lang, theme="monokai")
                    console.print(syntax)
                else:
                    console.print(content.text)
            elif content.type == "blob":
                console.print(f"[Binary data: {len(content.data)} bytes, {content.mimeType}]")
        
        return ""
        
    except Exception as e:
        return f"❌ Failed to read resource: {e}"


async def cmd_prompts(client: MCPClient, args: list[str]) -> str:
    """List available prompts."""
    server = args[0] if args else None
    
    try:
        prompts = await client.list_prompts(server)
    except Exception as e:
        return f"❌ Failed to list prompts: {e}"
    
    if not prompts:
        return "No prompts available"
    
    table = Table(title="Available Prompts")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white", max_width=50)
    table.add_column("Arguments", style="yellow")
    
    for prompt in prompts:
        arg_names = []
        if prompt.arguments:
            arg_names = [
                f"{'*' if a.required else ''}{a.name}" 
                for a in prompt.arguments
            ]
        
        table.add_row(
            prompt.name,
            (prompt.description or "")[:50],
            ", ".join(arg_names) or "none"
        )
    
    console.print(table)
    return f"\n{len(prompts)} prompt(s) available (* = required argument)"


async def cmd_prompt(client: MCPClient, args: list[str]) -> str:
    """Get a prompt."""
    if not args:
        return "Usage: prompt <name> [key=value ...]"
    
    prompt_name = args[0]
    
    # Parse arguments
    arguments = {}
    for arg in args[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            arguments[key] = value
    
    try:
        result = await client.get_prompt(prompt_name, arguments or None)
        
        console.print(f"\n[bold]{result.description}[/bold]\n")
        
        for message in result.messages:
            role_color = "blue" if message.role == "user" else "green"
            console.print(f"[{role_color}][{message.role}][/{role_color}]")
            
            if hasattr(message.content, 'text'):
                console.print(message.content.text)
            console.print()
        
        return ""
        
    except Exception as e:
        return f"❌ Failed to get prompt: {e}"


def cmd_help(client: MCPClient, args: list[str]) -> str:
    """Show help message."""
    help_text = """
[bold]MCP Client Commands[/bold]

[cyan]Connection:[/cyan]
  connect <name> <command> [args]  Connect to an MCP server
  disconnect [name]                Disconnect from a server
  servers                          List connected servers
  use <name>                       Set active server

[cyan]Tools:[/cyan]
  tools [server]                   List available tools
  tool <name>                      Show tool details
  call <tool> [args]               Call a tool
                                   Args: key=value or JSON

[cyan]Resources:[/cyan]
  resources [server]               List available resources
  read <uri>                       Read a resource

[cyan]Prompts:[/cyan]
  prompts [server]                 List available prompts
  prompt <name> [key=value]        Get a prompt

[cyan]General:[/cyan]
  help                             Show this help
  exit / quit                      Exit the client
"""
    console.print(help_text)
    return ""


# Command registry
COMMANDS = {
    "connect": cmd_connect,
    "disconnect": cmd_disconnect,
    "servers": cmd_servers,
    "use": cmd_use,
    "tools": cmd_tools,
    "tool": cmd_tool_info,
    "call": cmd_call,
    "resources": cmd_resources,
    "read": cmd_read,
    "prompts": cmd_prompts,
    "prompt": cmd_prompt,
    "help": cmd_help,
}
```

## Part 4: The REPL Interface

```python
# src/mcp_client/repl.py
"""REPL interface for the MCP client."""

import asyncio
import shlex
from rich.console import Console
from rich.prompt import Prompt

from .client import MCPClient
from .commands import COMMANDS, cmd_help

console = Console()


class REPL:
    """Interactive REPL for MCP client."""
    
    def __init__(self):
        self.client = MCPClient()
        self.running = True
    
    def get_prompt(self) -> str:
        """Generate the prompt string."""
        if self.client.active_server:
            return f"[bold cyan]mcp[/bold cyan]:[bold green]{self.client.active_server}[/bold green]> "
        return "[bold cyan]mcp[/bold cyan]> "
    
    async def execute_command(self, line: str) -> str:
        """Execute a command line."""
        line = line.strip()
        
        if not line:
            return ""
        
        # Parse command
        try:
            parts = shlex.split(line)
        except ValueError as e:
            return f"❌ Parse error: {e}"
        
        command = parts[0].lower()
        args = parts[1:]
        
        # Handle exit
        if command in ("exit", "quit", "q"):
            self.running = False
            return "Goodbye!"
        
        # Find and execute command handler
        if command in COMMANDS:
            handler = COMMANDS[command]
            if asyncio.iscoroutinefunction(handler):
                return await handler(self.client, args)
            else:
                return handler(self.client, args)
        else:
            return f"❌ Unknown command: {command}. Type 'help' for available commands."
    
    async def run(self):
        """Run the REPL loop."""
        console.print("\n[bold]MCP Interactive Client[/bold]")
        console.print("Type 'help' for available commands\n")
        
        while self.running:
            try:
                # Get input
                line = Prompt.ask(self.get_prompt())
                
                # Execute command
                result = await self.execute_command(line)
                
                if result:
                    console.print(result)
                
            except KeyboardInterrupt:
                console.print("\n[dim]Use 'exit' to quit[/dim]")
            
            except EOFError:
                self.running = False
                console.print("\nGoodbye!")
            
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        # Cleanup
        await self.client.disconnect_all()


async def main():
    """Entry point."""
    repl = REPL()
    await repl.run()


# src/mcp_client/__init__.py
import asyncio
from .repl import main as _main

def main():
    asyncio.run(_main())


# src/mcp_client/__main__.py
from . import main

if __name__ == "__main__":
    main()
```

## Part 5: Running the Client

### Installation

```bash
# Install dependencies
pip install mcp rich

# Or with the project
pip install -e .
```

### Usage Examples

```bash
# Start the client
mcp-client

# In the REPL:
mcp> help

# Connect to a server
mcp> connect files python -m file_ops_server
✅ Connected to file-ops-server v1.0.0

# List tools
mcp:files> tools
┌─────────────────────────────────────────────────────────┐
│                    Available Tools                       │
├──────────────┬──────────────────────────────────────────┤
│ Name         │ Description                               │
├──────────────┼──────────────────────────────────────────┤
│ create_file  │ Create a new file with the given content │
│ update_file  │ Update an existing file's content        │
│ delete_file  │ Delete a file from the workspace         │
│ search_files │ Search for files matching a glob pattern │
└──────────────┴──────────────────────────────────────────┘

# Call a tool
mcp:files> call create_file path=test.txt content="Hello World"
┌─────────────────────────────────────────────────────────┐
│                        Result                            │
├─────────────────────────────────────────────────────────┤
│ ✅ Created file: test.txt                                │
│ 📄 Size: 11 characters                                  │
└─────────────────────────────────────────────────────────┘

# Read a resource
mcp:files> read file://test.txt
Hello World

# Connect to another server
mcp:files> connect weather python -m weather_server
✅ Connected to weather-server v1.0.0

# Switch servers
mcp:weather> use files
✅ Now using server: files

# List all servers
mcp:files> servers
┌─────────────────────────────────────────────────────────┐
│                   Connected Servers                      │
├─────────┬────────────────────┬─────────┬────────────────┤
│ Name    │ Server             │ Version │ Active         │
├─────────┼────────────────────┼─────────┼────────────────┤
│ files   │ file-ops-server    │ 1.0.0   │ ✓              │
│ weather │ weather-server     │ 1.0.0   │                │
└─────────┴────────────────────┴─────────┴────────────────┘

# Exit
mcp:files> exit
Goodbye!
```

## Exercises

### Exercise 1: Add Command History
Implement command history using `readline` or a similar library.

### Exercise 2: Add Auto-completion
Add tab completion for commands, tool names, and resource URIs.

### Exercise 3: Add Output Formatting
Add a command to change output format (JSON, table, plain text).

### Exercise 4: Add Scripting Support
Add ability to run commands from a file.

## Summary

You've built a full-featured MCP client with:

- ✅ Multi-server connection management
- ✅ Interactive REPL interface
- ✅ Tool discovery and execution
- ✅ Resource browsing and reading
- ✅ Prompt listing and usage
- ✅ Beautiful terminal output with Rich

This client can be extended with additional features like scripting, history, and auto-completion.

Next: [Build a File System MCP Server Lab](/learn/mcp/mcp-clients/filesystem-lab) →
