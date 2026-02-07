# Defining Tool Schemas

Learn how to create effective tool schemas that enable LLMs to use your functions correctly.

## JSON Schema Fundamentals

Tool definitions use JSON Schema to describe function parameters. Understanding JSON Schema is essential.

### Basic JSON Schema Structure

```python
# A complete tool schema
tool_schema = {
    "type": "function",
    "function": {
        "name": "function_name",
        "description": "What this function does",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "Description of param1"
                },
                "param2": {
                    "type": "integer",
                    "description": "Description of param2"
                }
            },
            "required": ["param1"]
        }
    }
}
```

## JSON Schema Types

### String Type

```python
# Basic string
"name": {
    "type": "string",
    "description": "The user's full name"
}

# String with enum (fixed options)
"status": {
    "type": "string",
    "enum": ["pending", "active", "completed"],
    "description": "Current status of the task"
}

# String with pattern (regex)
"email": {
    "type": "string",
    "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$",
    "description": "Valid email address"
}

# String with length constraints
"username": {
    "type": "string",
    "minLength": 3,
    "maxLength": 20,
    "description": "Username (3-20 characters)"
}
```

### Number Types

```python
# Integer
"age": {
    "type": "integer",
    "minimum": 0,
    "maximum": 150,
    "description": "Person's age in years"
}

# Number (float)
"price": {
    "type": "number",
    "minimum": 0,
    "description": "Price in USD"
}

# Number with multiple of constraint
"quantity": {
    "type": "integer",
    "multipleOf": 5,
    "description": "Quantity (must be multiple of 5)"
}
```

### Boolean Type

```python
"is_active": {
    "type": "boolean",
    "description": "Whether the account is active"
}
```

### Array Type

```python
# Basic array of strings
"tags": {
    "type": "array",
    "items": {"type": "string"},
    "description": "List of tags"
}

# Array with constraints
"scores": {
    "type": "array",
    "items": {
        "type": "number",
        "minimum": 0,
        "maximum": 100
    },
    "minItems": 1,
    "maxItems": 10,
    "description": "List of scores (1-10 items)"
}

# Array of objects
"users": {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "email": {"type": "string"}
        },
        "required": ["name", "email"]
    },
    "description": "List of user objects"
}
```

### Object Type

```python
# Nested object
"address": {
    "type": "object",
    "properties": {
        "street": {"type": "string"},
        "city": {"type": "string"},
        "country": {"type": "string"},
        "postal_code": {"type": "string"}
    },
    "required": ["city", "country"],
    "description": "Physical address"
}
```

## Complete Tool Schema Examples

### 1. Web Search Tool

```python
web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information. Use this when you need to find current information, facts, or data that might not be in your training data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query. Be specific and include relevant keywords."
                },
                "num_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 5,
                    "description": "Number of results to return (1-10)"
                },
                "search_type": {
                    "type": "string",
                    "enum": ["web", "news", "images", "videos"],
                    "default": "web",
                    "description": "Type of search to perform"
                },
                "time_range": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year", "all"],
                    "default": "all",
                    "description": "Filter results by time range"
                }
            },
            "required": ["query"]
        }
    }
}
```

### 2. Database Query Tool

```python
database_query_tool = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": "Execute a read-only SQL query against the database. Only SELECT queries are allowed.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "SQL SELECT query to execute. Must be read-only."
                },
                "database": {
                    "type": "string",
                    "enum": ["users", "products", "orders", "analytics"],
                    "description": "Which database to query"
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 10,
                    "description": "Maximum number of rows to return"
                },
                "format": {
                    "type": "string",
                    "enum": ["table", "json", "csv"],
                    "default": "json",
                    "description": "Output format for the results"
                }
            },
            "required": ["query", "database"]
        }
    }
}
```

### 3. Email Tool

```python
email_tool = {
    "type": "function",
    "function": {
        "name": "send_email",
        "description": "Send an email to one or more recipients. Use this for communication tasks.",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "format": "email"
                    },
                    "minItems": 1,
                    "description": "List of recipient email addresses"
                },
                "cc": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of CC recipients (optional)"
                },
                "subject": {
                    "type": "string",
                    "maxLength": 200,
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content (plain text or HTML)"
                },
                "is_html": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether the body is HTML formatted"
                },
                "priority": {
                    "type": "string",
                    "enum": ["low", "normal", "high"],
                    "default": "normal",
                    "description": "Email priority level"
                },
                "attachments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "path": {"type": "string"}
                        },
                        "required": ["filename", "path"]
                    },
                    "description": "List of file attachments"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
}
```

### 4. Code Execution Tool

```python
code_execution_tool = {
    "type": "function",
    "function": {
        "name": "execute_code",
        "description": "Execute code in a sandboxed environment. Supports Python and JavaScript. Use this for calculations, data processing, or generating outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The code to execute"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript"],
                    "description": "Programming language"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 60,
                    "default": 30,
                    "description": "Maximum execution time in seconds"
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of packages/modules to import (e.g., ['pandas', 'numpy'])"
                }
            },
            "required": ["code", "language"]
        }
    }
}
```

### 5. File Operations Tool

```python
file_operations_tool = {
    "type": "function",
    "function": {
        "name": "file_operations",
        "description": "Perform file system operations: read, write, list, or delete files.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "delete", "exists"],
                    "description": "The operation to perform"
                },
                "path": {
                    "type": "string",
                    "description": "File or directory path"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for write operation)"
                },
                "encoding": {
                    "type": "string",
                    "enum": ["utf-8", "ascii", "latin-1"],
                    "default": "utf-8",
                    "description": "File encoding"
                },
                "recursive": {
                    "type": "boolean",
                    "default": False,
                    "description": "For list/delete: operate recursively"
                }
            },
            "required": ["operation", "path"]
        }
    }
}
```

## Writing Effective Descriptions

The description is crucial for LLM understanding:

```python
# ❌ BAD: Vague description
{
    "name": "search",
    "description": "Search for stuff"
}

# ✅ GOOD: Detailed description with examples
{
    "name": "web_search",
    "description": """Search the web for current information.

Use this tool when you need to:
- Find recent news or events
- Look up facts you're unsure about
- Get current prices, weather, or data
- Research topics in depth

Examples of good queries:
- "latest AI developments 2024"
- "current Bitcoin price USD"
- "population of Tokyo 2024"

Returns: A list of search results with titles, URLs, and snippets."""
}
```

### Description Best Practices

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Tool Description Best Practices                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. EXPLAIN THE PURPOSE                                                  │
│     "Use this tool to..." or "This tool allows..."                      │
│                                                                          │
│  2. SPECIFY WHEN TO USE IT                                              │
│     "Use when you need to find current information"                     │
│     "Use for calculations that require precision"                       │
│                                                                          │
│  3. INCLUDE EXAMPLES                                                     │
│     "Example: search('weather in Paris')"                               │
│     "Example queries: 'stock price AAPL', 'news about AI'"              │
│                                                                          │
│  4. MENTION LIMITATIONS                                                  │
│     "Only works with public URLs"                                       │
│     "Maximum file size: 10MB"                                           │
│                                                                          │
│  5. DESCRIBE THE OUTPUT                                                  │
│     "Returns: JSON object with {title, url, snippet}"                   │
│     "Returns: Plain text content of the file"                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Creating Schemas Programmatically

Use Python classes to generate schemas:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Literal, get_type_hints, get_origin, get_args
import inspect

def python_type_to_json_schema(python_type) -> dict:
    """Convert Python type hints to JSON Schema."""
    
    origin = get_origin(python_type)
    
    # Handle Optional
    if origin is type(None) or python_type is type(None):
        return {"type": "null"}
    
    # Handle Literal (enums)
    if origin is Literal:
        values = get_args(python_type)
        return {"type": "string", "enum": list(values)}
    
    # Handle List
    if origin is list:
        item_type = get_args(python_type)[0]
        return {
            "type": "array",
            "items": python_type_to_json_schema(item_type)
        }
    
    # Handle Optional
    if origin is type(Optional):
        inner_type = get_args(python_type)[0]
        return python_type_to_json_schema(inner_type)
    
    # Basic types
    type_mapping = {
        str: {"type": "string"},
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
    }
    
    return type_mapping.get(python_type, {"type": "string"})


def function_to_tool_schema(func, descriptions: dict = None) -> dict:
    """Convert a Python function to a tool schema."""
    
    descriptions = descriptions or {}
    hints = get_type_hints(func)
    sig = inspect.signature(func)
    
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        if name == "self":
            continue
            
        # Get type hint
        python_type = hints.get(name, str)
        
        # Convert to JSON Schema
        prop_schema = python_type_to_json_schema(python_type)
        
        # Add description if provided
        if name in descriptions:
            prop_schema["description"] = descriptions[name]
        
        # Add default if present
        if param.default is not inspect.Parameter.empty:
            prop_schema["default"] = param.default
        else:
            required.append(name)
        
        properties[name] = prop_schema
    
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }


# Example usage
def calculate_shipping(
    weight: float,
    destination: str,
    shipping_method: Literal["standard", "express", "overnight"],
    insurance: bool = False
) -> dict:
    """Calculate shipping cost for a package."""
    pass

# Generate schema automatically
schema = function_to_tool_schema(
    calculate_shipping,
    descriptions={
        "weight": "Package weight in kilograms",
        "destination": "Destination country code (e.g., 'US', 'UK')",
        "shipping_method": "Desired shipping speed",
        "insurance": "Whether to add shipping insurance"
    }
)

print(json.dumps(schema, indent=2))
```

## Pydantic Integration

Use Pydantic for robust schema generation:

```python
from pydantic import BaseModel, Field
from typing import Optional, List, Literal

class SearchParameters(BaseModel):
    """Parameters for web search."""
    
    query: str = Field(
        ...,
        description="Search query string",
        min_length=1,
        max_length=500
    )
    num_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    search_type: Literal["web", "news", "images"] = Field(
        default="web",
        description="Type of search"
    )
    safe_search: bool = Field(
        default=True,
        description="Enable safe search filtering"
    )

def pydantic_to_tool_schema(model: type[BaseModel], function_name: str) -> dict:
    """Convert Pydantic model to tool schema."""
    
    # Get JSON Schema from Pydantic
    json_schema = model.model_json_schema()
    
    # Extract properties and required fields
    properties = json_schema.get("properties", {})
    required = json_schema.get("required", [])
    
    return {
        "type": "function",
        "function": {
            "name": function_name,
            "description": model.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

# Generate schema from Pydantic model
search_schema = pydantic_to_tool_schema(SearchParameters, "web_search")
```

## Common Schema Patterns

### Optional Parameters with Defaults

```python
{
    "properties": {
        "limit": {
            "type": "integer",
            "default": 10,
            "description": "Optional, defaults to 10"
        }
    },
    "required": []  # Not in required list
}
```

### Mutually Exclusive Parameters

```python
# Use oneOf for mutually exclusive options
{
    "oneOf": [
        {
            "properties": {
                "user_id": {"type": "string"}
            },
            "required": ["user_id"]
        },
        {
            "properties": {
                "email": {"type": "string"}
            },
            "required": ["email"]
        }
    ]
}
```

### Conditional Parameters

```python
# If operation is "write", content is required
{
    "if": {
        "properties": {"operation": {"const": "write"}}
    },
    "then": {
        "required": ["content"]
    }
}
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Tool Schemas - Summary                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Schema Components:                                                      │
│    • name: Function identifier                                          │
│    • description: What it does and when to use it                       │
│    • parameters: JSON Schema defining inputs                            │
│                                                                          │
│  JSON Schema Types:                                                      │
│    string, integer, number, boolean, array, object                      │
│                                                                          │
│  Key Practices:                                                          │
│    • Write detailed descriptions with examples                          │
│    • Use enums for fixed options                                        │
│    • Mark required vs optional clearly                                  │
│    • Include constraints (min, max, pattern)                            │
│                                                                          │
│  Automation:                                                             │
│    • Generate from Python functions                                     │
│    • Use Pydantic for validation                                        │
│    • Create reusable schema templates                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Tool Execution](/learn/agents/tool-use/tool-execution) →
