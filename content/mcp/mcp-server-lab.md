# Lab: Build Your First MCP Server

Create a complete MCP server from scratch that exposes tools and resources for file operations.

## Lab Overview

```yaml
objectives:
  - "Build a fully functional MCP server"
  - "Implement tools for file operations"
  - "Implement resources for file access"
  - "Test with a simple client"

difficulty: Advanced
time: 30 minutes
prerequisites:
  - "MCP architecture understanding"
  - "Python async/await"
  - "Basic file operations"
```

## What We're Building

A **File Operations MCP Server** that provides:
- **Tools**: Create, update, and delete files
- **Resources**: Read files and list directories
- **Prompts**: Code review template for files

```
┌─────────────────────────────────────────────────────────────────┐
│                    File Operations MCP Server                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TOOLS:                                                          │
│    • create_file - Create a new file                            │
│    • update_file - Update file contents                         │
│    • delete_file - Delete a file                                │
│    • search_files - Search for files by pattern                 │
│                                                                  │
│  RESOURCES:                                                      │
│    • file://{path} - Read file contents                         │
│    • dir://{path} - List directory contents                     │
│                                                                  │
│  PROMPTS:                                                        │
│    • code_review - Review a file's code                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Part 1: Project Setup

### Step 1: Create Project Structure

```python
"""
Create this directory structure:

file_ops_server/
├── pyproject.toml
├── README.md
└── src/
    └── file_ops_server/
        ├── __init__.py
        ├── __main__.py
        ├── server.py
        ├── tools.py
        ├── resources.py
        ├── prompts.py
        └── config.py
"""
```

### Step 2: Configure pyproject.toml

```toml
[project]
name = "file-ops-server"
version = "1.0.0"
description = "MCP Server for file operations"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
]

[project.scripts]
file-ops-server = "file_ops_server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Step 3: Configuration Module

```python
# src/file_ops_server/config.py
"""Server configuration."""

import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Server configuration."""
    
    # Server info
    name: str = "file-ops-server"
    version: str = "1.0.0"
    
    # Security settings
    workspace_dir: Path = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: list[str] = None
    
    def __post_init__(self):
        if self.workspace_dir is None:
            self.workspace_dir = Path(os.getcwd())
        elif isinstance(self.workspace_dir, str):
            self.workspace_dir = Path(self.workspace_dir)
        
        if self.allowed_extensions is None:
            self.allowed_extensions = [
                ".txt", ".md", ".py", ".js", ".ts", 
                ".json", ".yaml", ".yml", ".html", ".css"
            ]
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        return cls(
            workspace_dir=Path(os.getenv("WORKSPACE_DIR", os.getcwd())),
            max_file_size=int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024)),
        )
    
    def is_path_allowed(self, path: Path) -> bool:
        """Check if a path is within the workspace."""
        try:
            resolved = path.resolve()
            resolved.relative_to(self.workspace_dir.resolve())
            return True
        except ValueError:
            return False
    
    def is_extension_allowed(self, path: Path) -> bool:
        """Check if file extension is allowed."""
        return path.suffix.lower() in self.allowed_extensions


# Global config instance
config = Config.from_env()
```

## Part 2: Implement Tools

```python
# src/file_ops_server/tools.py
"""File operation tools."""

import os
import glob
from pathlib import Path
from mcp.types import Tool, TextContent
from .config import config

# Tool definitions
TOOLS = [
    Tool(
        name="create_file",
        description="""Create a new file with the given content.
        
Will fail if the file already exists. Use update_file to modify existing files.
Path must be within the workspace directory.""",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path for the new file (e.g., 'src/main.py')"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            },
            "required": ["path", "content"]
        }
    ),
    Tool(
        name="update_file",
        description="""Update an existing file's content.
        
Will fail if the file doesn't exist. Use create_file for new files.""",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file"
                },
                "content": {
                    "type": "string",
                    "description": "New content for the file"
                },
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "default": "overwrite",
                    "description": "Write mode: overwrite or append"
                }
            },
            "required": ["path", "content"]
        }
    ),
    Tool(
        name="delete_file",
        description="Delete a file from the workspace.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file to delete"
                }
            },
            "required": ["path"]
        }
    ),
    Tool(
        name="search_files",
        description="""Search for files matching a glob pattern.
        
Examples:
- "*.py" - All Python files in current directory
- "**/*.py" - All Python files recursively
- "src/**/*.js" - All JS files under src/""",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to search for"
                },
                "include_content": {
                    "type": "boolean",
                    "default": False,
                    "description": "Include file contents in results"
                }
            },
            "required": ["pattern"]
        }
    )
]


def list_tools() -> list[Tool]:
    """Return available tools."""
    return TOOLS


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool."""
    
    if name == "create_file":
        return await _create_file(arguments)
    elif name == "update_file":
        return await _update_file(arguments)
    elif name == "delete_file":
        return await _delete_file(arguments)
    elif name == "search_files":
        return await _search_files(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _create_file(args: dict) -> list[TextContent]:
    """Create a new file."""
    rel_path = args["path"]
    content = args["content"]
    
    # Build full path
    file_path = config.workspace_dir / rel_path
    
    # Security checks
    if not config.is_path_allowed(file_path):
        return [TextContent(
            type="text",
            text=f"❌ Error: Path is outside workspace directory"
        )]
    
    if not config.is_extension_allowed(file_path):
        return [TextContent(
            type="text",
            text=f"❌ Error: File extension not allowed. Allowed: {config.allowed_extensions}"
        )]
    
    if file_path.exists():
        return [TextContent(
            type="text",
            text=f"❌ Error: File already exists: {rel_path}\nUse update_file to modify existing files."
        )]
    
    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write file
    file_path.write_text(content, encoding="utf-8")
    
    return [TextContent(
        type="text",
        text=f"✅ Created file: {rel_path}\n📄 Size: {len(content)} characters"
    )]


async def _update_file(args: dict) -> list[TextContent]:
    """Update an existing file."""
    rel_path = args["path"]
    content = args["content"]
    mode = args.get("mode", "overwrite")
    
    file_path = config.workspace_dir / rel_path
    
    # Security checks
    if not config.is_path_allowed(file_path):
        return [TextContent(
            type="text",
            text=f"❌ Error: Path is outside workspace directory"
        )]
    
    if not file_path.exists():
        return [TextContent(
            type="text",
            text=f"❌ Error: File not found: {rel_path}\nUse create_file for new files."
        )]
    
    # Write based on mode
    if mode == "append":
        existing = file_path.read_text(encoding="utf-8")
        file_path.write_text(existing + content, encoding="utf-8")
        action = "appended to"
    else:
        file_path.write_text(content, encoding="utf-8")
        action = "updated"
    
    return [TextContent(
        type="text",
        text=f"✅ Successfully {action}: {rel_path}"
    )]


async def _delete_file(args: dict) -> list[TextContent]:
    """Delete a file."""
    rel_path = args["path"]
    file_path = config.workspace_dir / rel_path
    
    # Security checks
    if not config.is_path_allowed(file_path):
        return [TextContent(
            type="text",
            text=f"❌ Error: Path is outside workspace directory"
        )]
    
    if not file_path.exists():
        return [TextContent(
            type="text",
            text=f"❌ Error: File not found: {rel_path}"
        )]
    
    if file_path.is_dir():
        return [TextContent(
            type="text",
            text=f"❌ Error: Cannot delete directory: {rel_path}"
        )]
    
    # Delete the file
    file_path.unlink()
    
    return [TextContent(
        type="text",
        text=f"✅ Deleted: {rel_path}"
    )]


async def _search_files(args: dict) -> list[TextContent]:
    """Search for files matching a pattern."""
    pattern = args["pattern"]
    include_content = args.get("include_content", False)
    
    # Search within workspace
    search_path = str(config.workspace_dir / pattern)
    matches = glob.glob(search_path, recursive=True)
    
    # Filter to files only and check security
    results = []
    for match in matches:
        path = Path(match)
        if path.is_file() and config.is_path_allowed(path):
            rel_path = path.relative_to(config.workspace_dir)
            
            result = {"path": str(rel_path), "size": path.stat().st_size}
            
            if include_content and config.is_extension_allowed(path):
                try:
                    result["content"] = path.read_text(encoding="utf-8")[:1000]
                    if len(path.read_text()) > 1000:
                        result["content"] += "\n... (truncated)"
                except Exception:
                    result["content"] = "(binary or unreadable)"
            
            results.append(result)
    
    if not results:
        return [TextContent(
            type="text",
            text=f"No files found matching: {pattern}"
        )]
    
    # Format output
    output = f"Found {len(results)} file(s) matching '{pattern}':\n\n"
    for r in results:
        output += f"📄 {r['path']} ({r['size']} bytes)\n"
        if "content" in r:
            output += f"```\n{r['content']}\n```\n\n"
    
    return [TextContent(type="text", text=output)]
```

## Part 3: Implement Resources

```python
# src/file_ops_server/resources.py
"""File resources."""

import os
from pathlib import Path
from mcp.types import Resource, ResourceTemplate, TextContent
from .config import config


def list_resources() -> list[Resource]:
    """List available file resources."""
    resources = []
    
    # Add workspace root as a resource
    resources.append(Resource(
        uri="dir://.",
        name="Workspace Root",
        description=f"Root directory: {config.workspace_dir}",
        mimeType="application/json"
    ))
    
    # Scan for files in workspace
    for file_path in config.workspace_dir.rglob("*"):
        if file_path.is_file() and config.is_extension_allowed(file_path):
            rel_path = file_path.relative_to(config.workspace_dir)
            
            # Determine MIME type
            mime_map = {
                ".txt": "text/plain",
                ".md": "text/markdown",
                ".py": "text/x-python",
                ".js": "text/javascript",
                ".ts": "text/typescript",
                ".json": "application/json",
                ".yaml": "text/yaml",
                ".yml": "text/yaml",
                ".html": "text/html",
                ".css": "text/css"
            }
            mime_type = mime_map.get(file_path.suffix.lower(), "text/plain")
            
            resources.append(Resource(
                uri=f"file://{rel_path}",
                name=file_path.name,
                description=f"File: {rel_path}",
                mimeType=mime_type
            ))
    
    return resources


def list_resource_templates() -> list[ResourceTemplate]:
    """List resource templates for dynamic access."""
    return [
        ResourceTemplate(
            uriTemplate="file://{path}",
            name="File Content",
            description="Read content of any file by path",
            mimeType="text/plain"
        ),
        ResourceTemplate(
            uriTemplate="dir://{path}",
            name="Directory Listing",
            description="List contents of any directory",
            mimeType="application/json"
        )
    ]


async def read_resource(uri: str) -> list[TextContent]:
    """Read a resource by URI."""
    
    if uri.startswith("file://"):
        return await _read_file_resource(uri)
    elif uri.startswith("dir://"):
        return await _read_dir_resource(uri)
    else:
        raise ValueError(f"Unknown resource scheme: {uri}")


async def _read_file_resource(uri: str) -> list[TextContent]:
    """Read a file resource."""
    rel_path = uri.replace("file://", "")
    file_path = config.workspace_dir / rel_path
    
    # Security check
    if not config.is_path_allowed(file_path):
        raise ValueError("Access denied: path outside workspace")
    
    if not file_path.exists():
        raise ValueError(f"File not found: {rel_path}")
    
    if not file_path.is_file():
        raise ValueError(f"Not a file: {rel_path}")
    
    # Check file size
    if file_path.stat().st_size > config.max_file_size:
        raise ValueError(f"File too large: {file_path.stat().st_size} bytes")
    
    # Read content
    content = file_path.read_text(encoding="utf-8")
    
    return [TextContent(type="text", text=content)]


async def _read_dir_resource(uri: str) -> list[TextContent]:
    """Read a directory listing resource."""
    import json
    
    rel_path = uri.replace("dir://", "")
    if rel_path == ".":
        dir_path = config.workspace_dir
    else:
        dir_path = config.workspace_dir / rel_path
    
    # Security check
    if not config.is_path_allowed(dir_path):
        raise ValueError("Access denied: path outside workspace")
    
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {rel_path}")
    
    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {rel_path}")
    
    # List contents
    entries = []
    for entry in sorted(dir_path.iterdir()):
        entry_rel = entry.relative_to(config.workspace_dir)
        entries.append({
            "name": entry.name,
            "path": str(entry_rel),
            "type": "directory" if entry.is_dir() else "file",
            "size": entry.stat().st_size if entry.is_file() else None
        })
    
    result = {
        "path": rel_path,
        "entries": entries,
        "total": len(entries)
    }
    
    return [TextContent(type="text", text=json.dumps(result, indent=2))]
```

## Part 4: Implement Prompts

```python
# src/file_ops_server/prompts.py
"""File-related prompts."""

from mcp.types import Prompt, PromptArgument, PromptMessage, TextContent


PROMPTS = [
    Prompt(
        name="code_review",
        description="Generate a code review for a file",
        arguments=[
            PromptArgument(
                name="file_path",
                description="Path to the file to review",
                required=True
            ),
            PromptArgument(
                name="focus",
                description="Review focus: security, performance, style, or all",
                required=False
            )
        ]
    ),
    Prompt(
        name="explain_code",
        description="Explain what a code file does",
        arguments=[
            PromptArgument(
                name="file_path",
                description="Path to the file to explain",
                required=True
            ),
            PromptArgument(
                name="detail_level",
                description="Level of detail: brief, standard, or detailed",
                required=False
            )
        ]
    )
]


def list_prompts() -> list[Prompt]:
    """Return available prompts."""
    return PROMPTS


async def get_prompt(name: str, arguments: dict | None = None) -> dict:
    """Get a prompt template with arguments filled in."""
    args = arguments or {}
    
    if name == "code_review":
        return _get_code_review_prompt(args)
    elif name == "explain_code":
        return _get_explain_code_prompt(args)
    else:
        raise ValueError(f"Unknown prompt: {name}")


def _get_code_review_prompt(args: dict) -> dict:
    """Generate code review prompt."""
    file_path = args.get("file_path", "")
    focus = args.get("focus", "all")
    
    focus_instructions = {
        "security": """Focus specifically on:
- Input validation vulnerabilities
- Authentication/authorization issues
- Data exposure risks
- Injection vulnerabilities""",
        "performance": """Focus specifically on:
- Algorithm efficiency
- Memory usage
- I/O operations
- Caching opportunities""",
        "style": """Focus specifically on:
- Code readability
- Naming conventions
- Documentation quality
- Code organization""",
        "all": """Review all aspects:
- Security vulnerabilities
- Performance issues
- Code style and readability
- Best practices"""
    }
    
    return {
        "description": f"Code review for {file_path}",
        "messages": [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Please review the code in the file at: {file_path}

{focus_instructions.get(focus, focus_instructions['all'])}

First, use the file resource to read the file content, then provide:

1. **Summary**: Brief description of what the code does
2. **Issues Found**: List any problems discovered
3. **Suggestions**: Specific improvements with code examples
4. **Rating**: Overall code quality (1-10)

Format your review in markdown."""
                )
            )
        ]
    }


def _get_explain_code_prompt(args: dict) -> dict:
    """Generate code explanation prompt."""
    file_path = args.get("file_path", "")
    detail_level = args.get("detail_level", "standard")
    
    detail_instructions = {
        "brief": "Provide a 2-3 sentence summary",
        "standard": "Explain the main components and how they work together",
        "detailed": "Provide a comprehensive explanation with examples"
    }
    
    return {
        "description": f"Explain code in {file_path}",
        "messages": [
            PromptMessage(
                role="user",
                content=TextContent(
                    type="text",
                    text=f"""Please explain the code in: {file_path}

Detail level: {detail_instructions.get(detail_level, detail_instructions['standard'])}

First, read the file using the file resource, then explain:
- What the code does
- Key functions/classes and their purposes
- How to use it
- Any important patterns or techniques used"""
                )
            )
        ]
    }
```

## Part 5: Main Server

```python
# src/file_ops_server/server.py
"""Main MCP server."""

import asyncio
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import config
from . import tools, resources, prompts

# Create server instance
server = Server(
    name=config.name,
    version=config.version
)

# Register tool handlers
@server.list_tools()
async def handle_list_tools():
    return tools.list_tools()

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict):
    return await tools.call_tool(name, arguments)

# Register resource handlers
@server.list_resources()
async def handle_list_resources():
    return resources.list_resources()

@server.list_resource_templates()
async def handle_list_resource_templates():
    return resources.list_resource_templates()

@server.read_resource()
async def handle_read_resource(uri: str):
    return await resources.read_resource(uri)

# Register prompt handlers
@server.list_prompts()
async def handle_list_prompts():
    return prompts.list_prompts()

@server.get_prompt()
async def handle_get_prompt(name: str, arguments: dict | None = None):
    return await prompts.get_prompt(name, arguments)


async def main():
    """Run the MCP server."""
    print(f"Starting {config.name} v{config.version}", file=__import__('sys').stderr)
    print(f"Workspace: {config.workspace_dir}", file=__import__('sys').stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


# src/file_ops_server/__init__.py
from .server import main

# src/file_ops_server/__main__.py
import asyncio
from . import main

if __name__ == "__main__":
    asyncio.run(main())
```

## Part 6: Testing Your Server

### Simple Test Client

```python
# test_client.py
"""Simple test client for the file ops server."""

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_server():
    """Test the file ops server."""
    
    # Start the server
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "file_ops_server"],
        env={"WORKSPACE_DIR": "./test_workspace"}
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize
            await session.initialize()
            print("✅ Server initialized\n")
            
            # Test list_tools
            print("📋 Available Tools:")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"  - {tool.name}: {tool.description[:50]}...")
            print()
            
            # Test list_resources
            print("📁 Available Resources:")
            resources = await session.list_resources()
            for resource in resources.resources[:5]:
                print(f"  - {resource.uri}: {resource.name}")
            print()
            
            # Test create_file
            print("🔧 Testing create_file:")
            result = await session.call_tool(
                "create_file",
                {
                    "path": "test_output/hello.txt",
                    "content": "Hello from MCP!"
                }
            )
            print(f"  {result.content[0].text}\n")
            
            # Test read_resource
            print("📖 Testing read_resource:")
            result = await session.read_resource("file://test_output/hello.txt")
            print(f"  Content: {result.contents[0].text}\n")
            
            # Test search_files
            print("🔍 Testing search_files:")
            result = await session.call_tool(
                "search_files",
                {"pattern": "**/*.txt"}
            )
            print(f"  {result.content[0].text}\n")
            
            # Test prompts
            print("💬 Available Prompts:")
            prompts = await session.list_prompts()
            for prompt in prompts.prompts:
                print(f"  - {prompt.name}: {prompt.description}")
            
            print("\n✅ All tests passed!")


if __name__ == "__main__":
    asyncio.run(test_server())
```

### Running the Server

```bash
# Install dependencies
pip install mcp

# Run the server directly
python -m file_ops_server

# Or with custom workspace
WORKSPACE_DIR=/path/to/workspace python -m file_ops_server

# Run tests
python test_client.py
```

### Configure for Claude Desktop

Add to Claude Desktop's `config.json`:

```json
{
  "mcpServers": {
    "file-ops": {
      "command": "python",
      "args": ["-m", "file_ops_server"],
      "env": {
        "WORKSPACE_DIR": "/path/to/your/workspace"
      }
    }
  }
}
```

## Exercises

### Exercise 1: Add a `move_file` Tool
Add a tool that moves/renames files within the workspace.

### Exercise 2: Add File Metadata Resource
Create a resource that returns file metadata (created date, modified date, permissions).

### Exercise 3: Add a `summarize_file` Prompt
Create a prompt template that asks the LLM to summarize a file's contents.

## Summary

Congratulations! You've built a complete MCP server with:

- ✅ **4 Tools**: create, update, delete, search files
- ✅ **2 Resource types**: file content and directory listings
- ✅ **2 Prompts**: code review and code explanation
- ✅ **Security**: Path validation and extension filtering
- ✅ **Configuration**: Environment-based settings

This server can be used with Claude Desktop, VS Code, or any MCP client to provide file system access to AI assistants.

Next: [MCP Client Implementation](/learn/mcp/mcp-clients/client-implementation) →
