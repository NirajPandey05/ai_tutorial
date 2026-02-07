# Implementing Tools in MCP Servers

A comprehensive guide to building powerful, well-designed tools for your MCP server.

## Tool Implementation Fundamentals

Tools are the primary way LLMs interact with external systems through MCP. Good tool design is crucial for effective AI applications.

```yaml
tool_design_principles:
  clarity: "Tools should have clear, descriptive names and documentation"
  atomicity: "Each tool should do one thing well"
  safety: "Tools should validate inputs and handle errors gracefully"
  feedback: "Tools should return informative results"
```

## Basic Tool Implementation

### Step 1: Define the Tool Schema

```python
from mcp.types import Tool

# Define a well-documented tool
weather_tool = Tool(
    name="get_weather",
    description="""Get the current weather for a location.

Returns temperature, conditions, humidity, and wind speed.
Supports cities worldwide. Use city name or 'city, country' format.

Examples:
- "San Francisco"
- "London, UK"
- "Tokyo, Japan"
""",
    inputSchema={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or 'city, country' format"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius",
                "description": "Temperature units"
            }
        },
        "required": ["location"]
    }
)
```

### Step 2: Implement the Handler

```python
from mcp.types import TextContent
import json

async def handle_get_weather(arguments: dict) -> list[TextContent]:
    """Handle the get_weather tool call."""
    
    # Extract and validate arguments
    location = arguments.get("location")
    if not location:
        return [TextContent(
            type="text",
            text="Error: Location is required"
        )]
    
    units = arguments.get("units", "celsius")
    
    # Call external API
    try:
        weather_data = await fetch_weather_api(location, units)
    except WeatherAPIError as e:
        return [TextContent(
            type="text",
            text=f"Could not fetch weather for '{location}': {str(e)}"
        )]
    
    # Format response
    result = f"""Weather for {weather_data['location']}:

🌡️ Temperature: {weather_data['temperature']}°{'C' if units == 'celsius' else 'F'}
☁️ Conditions: {weather_data['conditions']}
💧 Humidity: {weather_data['humidity']}%
💨 Wind: {weather_data['wind_speed']} km/h

Last updated: {weather_data['timestamp']}"""
    
    return [TextContent(type="text", text=result)]
```

### Step 3: Register with Server

```python
from mcp.server import Server

server = Server("weather-server")

@server.list_tools()
async def list_tools():
    return [weather_tool]

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        return await handle_get_weather(arguments)
    raise ValueError(f"Unknown tool: {name}")
```

## Input Validation

Robust input validation is essential for reliable tools:

```python
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class ValidationError:
    field: str
    message: str

class ToolInputValidator:
    """Validate tool inputs against schema."""
    
    def __init__(self, schema: dict):
        self.schema = schema
    
    def validate(self, arguments: dict) -> list[ValidationError]:
        """Validate arguments against schema."""
        errors = []
        properties = self.schema.get("properties", {})
        required = self.schema.get("required", [])
        
        # Check required fields
        for field in required:
            if field not in arguments:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing"
                ))
        
        # Validate each provided field
        for field, value in arguments.items():
            if field in properties:
                field_errors = self._validate_field(
                    field, value, properties[field]
                )
                errors.extend(field_errors)
        
        return errors
    
    def _validate_field(
        self, name: str, value: Any, schema: dict
    ) -> list[ValidationError]:
        """Validate a single field."""
        errors = []
        expected_type = schema.get("type")
        
        # Type validation
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict
        }
        
        if expected_type and expected_type in type_map:
            if not isinstance(value, type_map[expected_type]):
                errors.append(ValidationError(
                    field=name,
                    message=f"Expected {expected_type}, got {type(value).__name__}"
                ))
                return errors  # Skip further validation
        
        # Enum validation
        if "enum" in schema and value not in schema["enum"]:
            errors.append(ValidationError(
                field=name,
                message=f"Value must be one of: {schema['enum']}"
            ))
        
        # String constraints
        if expected_type == "string":
            if "minLength" in schema and len(value) < schema["minLength"]:
                errors.append(ValidationError(
                    field=name,
                    message=f"Minimum length is {schema['minLength']}"
                ))
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                errors.append(ValidationError(
                    field=name,
                    message=f"Maximum length is {schema['maxLength']}"
                ))
            if "pattern" in schema:
                import re
                if not re.match(schema["pattern"], value):
                    errors.append(ValidationError(
                        field=name,
                        message=f"Value must match pattern: {schema['pattern']}"
                    ))
        
        # Number constraints
        if expected_type in ("number", "integer"):
            if "minimum" in schema and value < schema["minimum"]:
                errors.append(ValidationError(
                    field=name,
                    message=f"Minimum value is {schema['minimum']}"
                ))
            if "maximum" in schema and value > schema["maximum"]:
                errors.append(ValidationError(
                    field=name,
                    message=f"Maximum value is {schema['maximum']}"
                ))
        
        return errors


# Using the validator
def validated_tool(name: str, description: str, schema: dict):
    """Decorator for tools with automatic validation."""
    validator = ToolInputValidator(schema)
    tool_def = Tool(name=name, description=description, inputSchema=schema)
    
    def decorator(func):
        async def wrapper(arguments: dict):
            # Validate inputs
            errors = validator.validate(arguments)
            if errors:
                error_messages = [f"- {e.field}: {e.message}" for e in errors]
                return [TextContent(
                    type="text",
                    text=f"Validation errors:\n" + "\n".join(error_messages)
                )]
            
            # Call the actual function
            return await func(arguments)
        
        wrapper.tool_definition = tool_def
        return wrapper
    
    return decorator


# Example usage
@validated_tool(
    name="send_email",
    description="Send an email",
    schema={
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "pattern": r"^[\w\.-]+@[\w\.-]+\.\w+$",
                "description": "Recipient email"
            },
            "subject": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
                "description": "Email subject"
            },
            "body": {
                "type": "string",
                "minLength": 1,
                "description": "Email body"
            }
        },
        "required": ["to", "subject", "body"]
    }
)
async def send_email(arguments: dict):
    # This function is only called if validation passes
    to = arguments["to"]
    subject = arguments["subject"]
    body = arguments["body"]
    
    # ... send email logic ...
    
    return [TextContent(type="text", text=f"Email sent to {to}")]
```

## Advanced Tool Patterns

### Tools with External APIs

```python
import httpx
from typing import Optional

class APIClient:
    """Wrapper for external API calls."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        headers = {}
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
    
    async def get(self, path: str, params: dict = None) -> dict:
        response = await self._client.get(path, params=params)
        response.raise_for_status()
        return response.json()
    
    async def post(self, path: str, data: dict) -> dict:
        response = await self._client.post(path, json=data)
        response.raise_for_status()
        return response.json()


# Tool using the API client
github_client = APIClient(
    base_url="https://api.github.com",
    api_key=os.getenv("GITHUB_TOKEN")
)

@validated_tool(
    name="github_create_issue",
    description="Create a GitHub issue in a repository",
    schema={
        "type": "object",
        "properties": {
            "repo": {
                "type": "string",
                "pattern": r"^[\w-]+/[\w-]+$",
                "description": "Repository in 'owner/repo' format"
            },
            "title": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
                "description": "Issue title"
            },
            "body": {
                "type": "string",
                "description": "Issue body (Markdown)"
            },
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Labels to apply"
            }
        },
        "required": ["repo", "title"]
    }
)
async def github_create_issue(arguments: dict):
    repo = arguments["repo"]
    
    async with github_client:
        try:
            result = await github_client.post(
                f"/repos/{repo}/issues",
                data={
                    "title": arguments["title"],
                    "body": arguments.get("body", ""),
                    "labels": arguments.get("labels", [])
                }
            )
            
            return [TextContent(
                type="text",
                text=f"✅ Issue created: {result['html_url']}\n\n"
                     f"Issue #{result['number']}: {result['title']}"
            )]
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return [TextContent(
                    type="text",
                    text=f"❌ Repository '{repo}' not found or no access"
                )]
            elif e.response.status_code == 401:
                return [TextContent(
                    type="text",
                    text="❌ Authentication failed. Check your GitHub token."
                )]
            else:
                return [TextContent(
                    type="text",
                    text=f"❌ GitHub API error: {e.response.status_code}"
                )]
```

### Tools with State

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

@dataclass
class ConversationState:
    """State for a conversational tool."""
    history: List[dict] = field(default_factory=list)
    context: Dict[str, any] = field(default_factory=dict)
    last_active: datetime = field(default_factory=datetime.now)

class StatefulToolManager:
    """Manager for tools that maintain state."""
    
    def __init__(self):
        self._sessions: Dict[str, ConversationState] = {}
    
    def get_session(self, session_id: str) -> ConversationState:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState()
        
        session = self._sessions[session_id]
        session.last_active = datetime.now()
        return session
    
    def clear_session(self, session_id: str):
        if session_id in self._sessions:
            del self._sessions[session_id]

state_manager = StatefulToolManager()

@validated_tool(
    name="remember",
    description="Store information for later recall in the current session",
    schema={
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Session identifier"},
            "key": {"type": "string", "description": "What to remember"},
            "value": {"type": "string", "description": "Information to store"}
        },
        "required": ["session_id", "key", "value"]
    }
)
async def remember_tool(arguments: dict):
    session = state_manager.get_session(arguments["session_id"])
    session.context[arguments["key"]] = arguments["value"]
    
    return [TextContent(
        type="text",
        text=f"✓ Remembered: {arguments['key']}"
    )]

@validated_tool(
    name="recall",
    description="Recall previously stored information",
    schema={
        "type": "object",
        "properties": {
            "session_id": {"type": "string"},
            "key": {"type": "string", "description": "What to recall"}
        },
        "required": ["session_id", "key"]
    }
)
async def recall_tool(arguments: dict):
    session = state_manager.get_session(arguments["session_id"])
    key = arguments["key"]
    
    if key in session.context:
        return [TextContent(
            type="text",
            text=f"{key}: {session.context[key]}"
        )]
    else:
        return [TextContent(
            type="text",
            text=f"No information stored for '{key}'"
        )]
```

### Tools with Progress Reporting

```python
from typing import AsyncIterator

async def long_running_operation(
    items: list,
    progress_callback: callable = None
) -> list:
    """Simulate a long-running operation with progress."""
    results = []
    total = len(items)
    
    for i, item in enumerate(items):
        # Process item
        result = await process_item(item)
        results.append(result)
        
        # Report progress
        if progress_callback:
            await progress_callback(i + 1, total)
    
    return results

@validated_tool(
    name="batch_process",
    description="Process multiple items with progress tracking",
    schema={
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Items to process"
            }
        },
        "required": ["items"]
    }
)
async def batch_process_tool(arguments: dict):
    items = arguments["items"]
    progress_messages = []
    
    async def track_progress(current: int, total: int):
        percent = (current / total) * 100
        progress_messages.append(f"Progress: {current}/{total} ({percent:.0f}%)")
    
    # Run the operation
    results = await long_running_operation(items, track_progress)
    
    # Format results
    output = "## Processing Complete\n\n"
    output += f"Processed {len(results)} items.\n\n"
    output += "### Results:\n"
    for i, result in enumerate(results):
        output += f"- Item {i+1}: {result}\n"
    
    return [TextContent(type="text", text=output)]
```

## Tool Categories

Organize tools by category for better discoverability:

```python
from enum import Enum
from dataclasses import dataclass

class ToolCategory(Enum):
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    API = "api"
    COMMUNICATION = "communication"
    ANALYSIS = "analysis"
    UTILITY = "utility"

@dataclass
class CategorizedTool:
    tool: Tool
    category: ToolCategory
    handler: callable

class CategorizedToolRegistry:
    """Registry with tool categorization."""
    
    def __init__(self):
        self._tools: Dict[str, CategorizedTool] = {}
    
    def register(
        self,
        name: str,
        description: str,
        schema: dict,
        category: ToolCategory
    ):
        def decorator(func):
            self._tools[name] = CategorizedTool(
                tool=Tool(name=name, description=description, inputSchema=schema),
                category=category,
                handler=func
            )
            return func
        return decorator
    
    def list_tools(self) -> list[Tool]:
        return [t.tool for t in self._tools.values()]
    
    def list_by_category(self, category: ToolCategory) -> list[Tool]:
        return [
            t.tool for t in self._tools.values()
            if t.category == category
        ]
    
    async def call_tool(self, name: str, arguments: dict):
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return await self._tools[name].handler(arguments)

# Usage
registry = CategorizedToolRegistry()

@registry.register(
    name="list_files",
    description="List files in a directory",
    schema={"type": "object", "properties": {"path": {"type": "string"}}},
    category=ToolCategory.FILE_SYSTEM
)
async def list_files(args):
    # ...
    pass

@registry.register(
    name="send_slack",
    description="Send a Slack message",
    schema={"type": "object", "properties": {"channel": {"type": "string"}, "message": {"type": "string"}}},
    category=ToolCategory.COMMUNICATION
)
async def send_slack(args):
    # ...
    pass
```

## Best Practices Summary

```yaml
tool_best_practices:
  naming:
    - "Use clear, action-oriented names (verb_noun)"
    - "Be specific: 'search_github_repos' not 'search'"
    - "Use consistent naming conventions"
  
  descriptions:
    - "Write for the LLM that will read them"
    - "Include examples and use cases"
    - "Document limitations and edge cases"
    - "Specify expected output format"
  
  schemas:
    - "Use JSON Schema validation"
    - "Provide default values where sensible"
    - "Include constraints (min, max, patterns)"
    - "Make descriptions clear and helpful"
  
  implementation:
    - "Validate all inputs"
    - "Handle errors gracefully"
    - "Return informative results"
    - "Use timeouts for external calls"
    - "Log operations for debugging"
  
  security:
    - "Sanitize user inputs"
    - "Validate file paths and URLs"
    - "Implement rate limiting"
    - "Use least privilege principle"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│               Implementing Tools - Summary                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tool Components:                                                │
│    • Name: Clear, action-oriented identifier                    │
│    • Description: Detailed documentation for LLMs               │
│    • Schema: JSON Schema for input validation                   │
│    • Handler: Async function that executes the tool             │
│                                                                  │
│  Key Patterns:                                                   │
│    • Input validation with schema                               │
│    • Error handling with informative messages                   │
│    • External API integration                                   │
│    • Stateful operations when needed                            │
│    • Progress tracking for long operations                      │
│                                                                  │
│  Remember:                                                       │
│    • Tools are for ACTIONS with side effects                    │
│    • Write descriptions for LLMs, not humans                    │
│    • Validate everything, trust nothing                         │
│    • Return structured, parseable results                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [Implementing Resources](/learn/mcp/mcp-servers/implementing-resources) →
