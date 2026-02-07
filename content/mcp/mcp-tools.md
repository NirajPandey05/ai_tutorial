# Tools in MCP

Tools are executable functions that MCP servers expose to clients, enabling LLMs to perform actions in the real world.

## What are Tools?

Tools allow LLMs to **take actions** beyond generating text. Unlike resources (read-only data), tools execute operations that can have side effects.

```yaml
tools_overview:
  definition: "Executable functions exposed by MCP servers"
  
  characteristics:
    - "Can have side effects"
    - "Accept structured input parameters"
    - "Return execution results"
    - "Defined with JSON Schema"
  
  examples:
    - "send_email(to, subject, body)"
    - "create_github_issue(repo, title, body)"
    - "run_sql_query(query)"
    - "search_files(pattern, directory)"
```

```
┌─────────────────────────────────────────────────────────────────┐
│                   How Tools Work in MCP                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Server declares tools with schemas                         │
│   2. Client retrieves available tools                           │
│   3. LLM decides when to use a tool                             │
│   4. Client sends tool call request                             │
│   5. Server executes and returns result                         │
│   6. LLM incorporates result into response                      │
│                                                                  │
│   User                                                           │
│     │                                                            │
│     ▼                                                            │
│   ┌───────────┐    "List tools"    ┌───────────┐                │
│   │   HOST    │◄──────────────────►│   MCP     │                │
│   │  (LLM +   │    Tool schemas    │  SERVER   │                │
│   │  Client)  │                    │           │                │
│   │           │    "Call tool X"   │           │                │
│   │           │───────────────────►│  Execute  │                │
│   │           │◄───────────────────│  Return   │                │
│   └───────────┘    Result          └───────────┘                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Defining Tools

Tools are defined with a name, description, and JSON Schema for input parameters:

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

server = Server("tool-demo")

@server.list_tools()
async def list_tools():
    """Define available tools."""
    return [
        Tool(
            name="send_email",
            description="Send an email to a recipient",
            inputSchema={
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Recipient email address"
                    },
                    "subject": {
                        "type": "string",
                        "description": "Email subject line"
                    },
                    "body": {
                        "type": "string",
                        "description": "Email body content"
                    },
                    "cc": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "CC recipients (optional)"
                    }
                },
                "required": ["to", "subject", "body"]
            }
        )
    ]
```

### Input Schema Best Practices

```python
# Well-designed tool schema
Tool(
    name="search_database",
    description="""Search the product database.
    
    Returns matching products with name, price, and availability.
    Use specific terms for better results.
    Maximum 100 results per query.""",
    
    inputSchema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms (e.g., 'red shoes size 10')",
                "minLength": 1,
                "maxLength": 200
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "books", "home"],
                "description": "Filter by product category"
            },
            "min_price": {
                "type": "number",
                "minimum": 0,
                "description": "Minimum price filter"
            },
            "max_price": {
                "type": "number",
                "minimum": 0,
                "description": "Maximum price filter"
            },
            "in_stock_only": {
                "type": "boolean",
                "default": False,
                "description": "Only show items in stock"
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "default": 10,
                "description": "Maximum results to return"
            }
        },
        "required": ["query"]
    }
)
```

## Implementing Tools

### Basic Tool Implementation

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict):
    """Execute a tool call."""
    
    if name == "send_email":
        # Validate and extract arguments
        to = arguments["to"]
        subject = arguments["subject"]
        body = arguments["body"]
        cc = arguments.get("cc", [])
        
        # Execute the action
        try:
            result = await send_email_impl(to, subject, body, cc)
            return [TextContent(
                type="text",
                text=f"Email sent successfully to {to}"
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Failed to send email: {str(e)}"
            )]
    
    raise ValueError(f"Unknown tool: {name}")
```

### Multiple Tools Pattern

```python
from typing import Callable, Any
from dataclasses import dataclass

@dataclass
class ToolDefinition:
    """Tool definition with metadata and implementation."""
    tool: Tool
    handler: Callable

class ToolRegistry:
    """Registry for managing multiple tools."""
    
    def __init__(self):
        self.tools: dict[str, ToolDefinition] = {}
    
    def register(self, name: str, description: str, schema: dict):
        """Decorator to register a tool."""
        def decorator(func: Callable):
            self.tools[name] = ToolDefinition(
                tool=Tool(
                    name=name,
                    description=description,
                    inputSchema=schema
                ),
                handler=func
            )
            return func
        return decorator
    
    def list_tools(self) -> list[Tool]:
        return [td.tool for td in self.tools.values()]
    
    async def call_tool(self, name: str, arguments: dict) -> Any:
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")
        return await self.tools[name].handler(arguments)

# Usage
registry = ToolRegistry()

@registry.register(
    name="get_weather",
    description="Get current weather for a location",
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    }
)
async def get_weather(args: dict):
    city = args["city"]
    units = args.get("units", "celsius")
    weather = await fetch_weather(city, units)
    return [TextContent(type="text", text=json.dumps(weather))]

@registry.register(
    name="search_web",
    description="Search the web for information",
    schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "default": 5}
        },
        "required": ["query"]
    }
)
async def search_web(args: dict):
    results = await web_search(args["query"], args.get("num_results", 5))
    return [TextContent(type="text", text=json.dumps(results))]

# Connect to server
@server.list_tools()
async def list_tools():
    return registry.list_tools()

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    return await registry.call_tool(name, arguments)
```

## Tool Return Types

Tools can return different content types:

### Text Results

```python
# Simple text response
return [TextContent(
    type="text",
    text="Operation completed successfully"
)]

# Structured data as JSON text
return [TextContent(
    type="text",
    text=json.dumps({
        "status": "success",
        "data": {"id": 123, "name": "Created item"}
    }, indent=2)
)]

# Multiple text blocks
return [
    TextContent(type="text", text="## Summary\nOperation complete"),
    TextContent(type="text", text="## Details\n- Step 1: Done\n- Step 2: Done")
]
```

### Image Results

```python
from mcp.types import ImageContent
import base64

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "generate_chart":
        # Generate chart image
        chart_bytes = generate_chart(arguments)
        
        return [ImageContent(
            type="image",
            data=base64.b64encode(chart_bytes).decode(),
            mimeType="image/png"
        )]
```

### Embedded Resources

```python
from mcp.types import EmbeddedResource, Resource

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "create_file":
        # Create file and return as embedded resource
        file_path = create_file(arguments["name"], arguments["content"])
        
        return [
            TextContent(type="text", text=f"Created file: {file_path}"),
            EmbeddedResource(
                type="resource",
                resource=Resource(
                    uri=f"file://{file_path}",
                    name=arguments["name"],
                    mimeType="text/plain"
                )
            )
        ]
```

## Error Handling

Proper error handling is crucial for tools:

```python
from mcp.types import TextContent

class ToolError(Exception):
    """Custom tool error with user-friendly message."""
    pass

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    try:
        if name == "database_query":
            return await execute_query(arguments)
            
    except ValidationError as e:
        # Input validation failed
        return [TextContent(
            type="text",
            text=f"Invalid input: {e.message}"
        )]
        
    except PermissionError as e:
        # Access denied
        return [TextContent(
            type="text",
            text=f"Access denied: {str(e)}"
        )]
        
    except RateLimitError as e:
        # Rate limited
        return [TextContent(
            type="text",
            text=f"Rate limit exceeded. Try again in {e.retry_after} seconds."
        )]
        
    except Exception as e:
        # Unexpected error - log but return safe message
        logger.exception(f"Tool error: {name}")
        return [TextContent(
            type="text",
            text=f"An error occurred while executing {name}. Please try again."
        )]
    
    raise ValueError(f"Unknown tool: {name}")
```

## Real-World Tool Examples

### GitHub Integration Tools

```python
tools = [
    Tool(
        name="github_create_issue",
        description="Create a new GitHub issue",
        inputSchema={
            "type": "object",
            "properties": {
                "repo": {
                    "type": "string",
                    "description": "Repository in format 'owner/repo'"
                },
                "title": {
                    "type": "string",
                    "description": "Issue title"
                },
                "body": {
                    "type": "string",
                    "description": "Issue description (Markdown supported)"
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Labels to apply"
                },
                "assignees": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Users to assign"
                }
            },
            "required": ["repo", "title"]
        }
    ),
    Tool(
        name="github_list_prs",
        description="List open pull requests for a repository",
        inputSchema={
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "state": {"type": "string", "enum": ["open", "closed", "all"]},
                "limit": {"type": "integer", "default": 10}
            },
            "required": ["repo"]
        }
    ),
    Tool(
        name="github_search_code",
        description="Search for code across repositories",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "repo": {"type": "string", "description": "Limit to specific repo"},
                "language": {"type": "string", "description": "Filter by language"}
            },
            "required": ["query"]
        }
    )
]
```

### Database Tools

```python
tools = [
    Tool(
        name="db_query",
        description="""Execute a read-only SQL query.
        Only SELECT statements are allowed.
        Returns up to 100 rows.""",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query"
                },
                "params": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Query parameters (for ? placeholders)"
                }
            },
            "required": ["query"]
        }
    ),
    Tool(
        name="db_describe_table",
        description="Get schema information for a database table",
        inputSchema={
            "type": "object",
            "properties": {
                "table_name": {"type": "string"}
            },
            "required": ["table_name"]
        }
    ),
    Tool(
        name="db_list_tables",
        description="List all tables in the database",
        inputSchema={
            "type": "object",
            "properties": {}
        }
    )
]
```

### File System Tools

```python
tools = [
    Tool(
        name="read_file",
        description="Read contents of a file",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path"},
                "encoding": {"type": "string", "default": "utf-8"}
            },
            "required": ["path"]
        }
    ),
    Tool(
        name="write_file",
        description="Write content to a file",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["overwrite", "append"],
                    "default": "overwrite"
                }
            },
            "required": ["path", "content"]
        }
    ),
    Tool(
        name="search_files",
        description="Search for files matching a pattern",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern (e.g., '**/*.py')"
                },
                "directory": {
                    "type": "string",
                    "description": "Starting directory",
                    "default": "."
                },
                "include_content": {
                    "type": "boolean",
                    "description": "Include file contents in results",
                    "default": False
                }
            },
            "required": ["pattern"]
        }
    )
]
```

## Tool Annotations

Add metadata to help LLMs use tools effectively:

```python
Tool(
    name="execute_code",
    description="""Execute Python code in a sandboxed environment.

    IMPORTANT: This tool runs arbitrary code. Only use for:
    - Data analysis
    - Mathematical calculations
    - Text processing

    DO NOT use for:
    - File system modifications
    - Network requests
    - System commands
    
    Timeout: 30 seconds
    Memory limit: 256MB""",
    
    inputSchema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            }
        },
        "required": ["code"]
    },
    
    # Additional annotations (MCP extension)
    annotations={
        "readOnly": False,
        "destructive": False,
        "requiresConfirmation": True,
        "timeout": 30000,
        "category": "code_execution"
    }
)
```

## Client Usage

### Calling Tools from Client

```python
from mcp import ClientSession

async def use_tools(session: ClientSession):
    # List available tools
    tools = await session.list_tools()
    print(f"Available tools: {[t.name for t in tools.tools]}")
    
    # Call a tool
    result = await session.call_tool(
        name="search_files",
        arguments={
            "pattern": "**/*.py",
            "directory": "./src"
        }
    )
    
    # Process result
    for content in result.content:
        if content.type == "text":
            print(f"Result: {content.text}")
```

### Integration with LLM

```python
async def chat_with_tools(user_message: str, session: ClientSession):
    # Get available tools
    mcp_tools = await session.list_tools()
    
    # Convert to OpenAI format
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        }
        for tool in mcp_tools.tools
    ]
    
    # Call LLM with tools
    response = await openai.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": user_message}],
        tools=openai_tools
    )
    
    # Execute any tool calls
    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            result = await session.call_tool(
                name=tool_call.function.name,
                arguments=json.loads(tool_call.function.arguments)
            )
            # Continue conversation with result...
    
    return response
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                       MCP Tools Summary                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tools = Executable functions that perform actions              │
│                                                                  │
│  Components:                                                     │
│    • name: Unique identifier                                    │
│    • description: What the tool does (LLM reads this!)          │
│    • inputSchema: JSON Schema for parameters                    │
│                                                                  │
│  Implementation:                                                 │
│    • list_tools(): Declare available tools                      │
│    • call_tool(): Execute tool with arguments                   │
│                                                                  │
│  Return Types:                                                   │
│    • TextContent: Text/JSON responses                           │
│    • ImageContent: Images with base64 data                      │
│    • EmbeddedResource: References to resources                  │
│                                                                  │
│  Best Practices:                                                 │
│    • Write detailed descriptions                                │
│    • Use proper JSON Schema validation                          │
│    • Handle errors gracefully                                   │
│    • Return structured, parseable results                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [Prompts in MCP](/learn/mcp/mcp-core-concepts/prompts) →
