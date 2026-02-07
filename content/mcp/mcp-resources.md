# Resources in MCP

Resources are data sources that MCP servers expose to clients, allowing LLMs to read contextual information.

## What are Resources?

Resources represent **readable data** that an LLM can access through MCP. Unlike tools (which perform actions), resources provide information that can be included in the LLM's context.

```yaml
resources_overview:
  definition: "URI-addressable data sources exposed by MCP servers"
  
  characteristics:
    - "Read-only (tools are for actions)"
    - "URI-based addressing"
    - "Support various content types"
    - "Can be subscribed to for updates"
  
  examples:
    - "File contents (file://)"
    - "Database records (db://)"
    - "API responses (api://)"
    - "Configuration (config://)"
    - "Documentation (docs://)"
```

```
┌─────────────────────────────────────────────────────────────────┐
│                    Tools vs Resources                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TOOLS                           RESOURCES                      │
│   ─────                           ─────────                      │
│   • Execute actions               • Provide data                 │
│   • Have side effects             • Read-only                    │
│   • Called with parameters        • Addressed by URI             │
│   • Return results                • Return content               │
│                                                                  │
│   Examples:                       Examples:                      │
│   • send_email()                  • file://config.json           │
│   • create_issue()                • db://users/123               │
│   • run_query()                   • docs://api/reference         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Resource Structure

Every resource has these key properties:

```python
from mcp.types import Resource

resource = Resource(
    uri="file:///project/README.md",        # Unique identifier
    name="Project README",                   # Human-readable name
    description="Project documentation",     # What this resource contains
    mimeType="text/markdown"                 # Content type
)
```

### Resource URI Schemes

```yaml
common_uri_schemes:
  file:
    pattern: "file:///path/to/file"
    example: "file:///home/user/project/config.json"
    use_case: "Local file system access"
    
  db:
    pattern: "db://table/id"
    example: "db://users/42"
    use_case: "Database records"
    
  api:
    pattern: "api://service/endpoint"
    example: "api://github/repos/user/repo"
    use_case: "External API data"
    
  config:
    pattern: "config://name"
    example: "config://app-settings"
    use_case: "Configuration data"
    
  custom:
    pattern: "yourscheme://path"
    example: "myapp://dashboard/metrics"
    use_case: "Application-specific resources"
```

## Implementing Resources

### Basic Resource Server

```python
from mcp.server import Server
from mcp.types import Resource, TextContent, BlobContent

server = Server("resource-demo")

# In-memory data store (could be files, database, API, etc.)
DOCUMENTS = {
    "readme": {
        "content": "# My Project\n\nWelcome to my project!",
        "mime": "text/markdown"
    },
    "config": {
        "content": '{"debug": true, "version": "1.0.0"}',
        "mime": "application/json"
    }
}

@server.list_resources()
async def list_resources():
    """List all available resources."""
    return [
        Resource(
            uri=f"docs://{doc_id}",
            name=f"Document: {doc_id}",
            description=f"Content of {doc_id}",
            mimeType=data["mime"]
        )
        for doc_id, data in DOCUMENTS.items()
    ]

@server.read_resource()
async def read_resource(uri: str):
    """Read a specific resource by URI."""
    # Parse the URI
    if uri.startswith("docs://"):
        doc_id = uri.replace("docs://", "")
        
        if doc_id in DOCUMENTS:
            doc = DOCUMENTS[doc_id]
            return [TextContent(
                type="text",
                text=doc["content"]
            )]
    
    raise ValueError(f"Resource not found: {uri}")
```

### File System Resources

```python
import os
from pathlib import Path

server = Server("filesystem-resources")

ALLOWED_DIR = Path("/path/to/allowed/directory")

@server.list_resources()
async def list_resources():
    """List files as resources."""
    resources = []
    
    for file_path in ALLOWED_DIR.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(ALLOWED_DIR)
            mime_type = guess_mime_type(file_path)
            
            resources.append(Resource(
                uri=f"file://{rel_path}",
                name=file_path.name,
                description=f"File: {rel_path}",
                mimeType=mime_type
            ))
    
    return resources

@server.read_resource()
async def read_resource(uri: str):
    """Read file content."""
    if not uri.startswith("file://"):
        raise ValueError(f"Unknown URI scheme: {uri}")
    
    rel_path = uri.replace("file://", "")
    file_path = ALLOWED_DIR / rel_path
    
    # Security: Ensure path is within allowed directory
    if not file_path.resolve().is_relative_to(ALLOWED_DIR.resolve()):
        raise ValueError("Access denied: Path outside allowed directory")
    
    if not file_path.exists():
        raise ValueError(f"File not found: {rel_path}")
    
    # Read based on type
    mime_type = guess_mime_type(file_path)
    
    if mime_type.startswith("text/"):
        content = file_path.read_text()
        return [TextContent(type="text", text=content)]
    else:
        content = file_path.read_bytes()
        return [BlobContent(
            type="blob",
            data=base64.b64encode(content).decode(),
            mimeType=mime_type
        )]


def guess_mime_type(path: Path) -> str:
    """Guess MIME type from file extension."""
    suffix = path.suffix.lower()
    mime_map = {
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".json": "application/json",
        ".py": "text/x-python",
        ".js": "text/javascript",
        ".html": "text/html",
        ".css": "text/css",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".pdf": "application/pdf",
    }
    return mime_map.get(suffix, "application/octet-stream")
```

### Database Resources

```python
import sqlite3

server = Server("database-resources")

DB_PATH = "app.db"

@server.list_resources()
async def list_resources():
    """List database tables as resources."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    resources = []
    for (table_name,) in tables:
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        
        resources.append(Resource(
            uri=f"db://{table_name}",
            name=f"Table: {table_name}",
            description=f"Database table with {count} rows",
            mimeType="application/json"
        ))
    
    conn.close()
    return resources

@server.read_resource()
async def read_resource(uri: str):
    """Read database table content."""
    if not uri.startswith("db://"):
        raise ValueError(f"Unknown URI scheme: {uri}")
    
    table_name = uri.replace("db://", "")
    
    # Security: Validate table name (prevent SQL injection)
    if not table_name.isalnum():
        raise ValueError("Invalid table name")
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
        rows = [dict(row) for row in cursor.fetchall()]
        
        return [TextContent(
            type="text",
            text=json.dumps(rows, indent=2, default=str)
        )]
    finally:
        conn.close()
```

## Resource Content Types

MCP supports different content types for resources:

### Text Content

```python
from mcp.types import TextContent

# Plain text, markdown, code, JSON as text
text_content = TextContent(
    type="text",
    text="# Hello World\n\nThis is markdown content."
)
```

### Blob Content (Binary)

```python
from mcp.types import BlobContent
import base64

# Images, PDFs, binary files
with open("image.png", "rb") as f:
    image_data = f.read()

blob_content = BlobContent(
    type="blob",
    data=base64.b64encode(image_data).decode(),
    mimeType="image/png"
)
```

### Embedded Resources

```python
from mcp.types import EmbeddedResource

# Resource embedded within another resource
embedded = EmbeddedResource(
    type="resource",
    resource=Resource(
        uri="file://nested/config.json",
        name="Nested Config",
        mimeType="application/json"
    )
)
```

## Resource Templates

For dynamic resources, use URI templates:

```python
from mcp.types import ResourceTemplate

@server.list_resource_templates()
async def list_resource_templates():
    """List resource templates for dynamic URIs."""
    return [
        ResourceTemplate(
            uriTemplate="db://users/{user_id}",
            name="User Profile",
            description="Get a specific user by ID",
            mimeType="application/json"
        ),
        ResourceTemplate(
            uriTemplate="api://weather/{city}",
            name="Weather Data",
            description="Get weather for a city",
            mimeType="application/json"
        )
    ]

@server.read_resource()
async def read_resource(uri: str):
    """Read resource, handling templates."""
    # Handle user profile template
    if uri.startswith("db://users/"):
        user_id = uri.replace("db://users/", "")
        user = get_user(user_id)
        return [TextContent(type="text", text=json.dumps(user))]
    
    # Handle weather template
    if uri.startswith("api://weather/"):
        city = uri.replace("api://weather/", "")
        weather = get_weather(city)
        return [TextContent(type="text", text=json.dumps(weather))]
    
    raise ValueError(f"Unknown resource: {uri}")
```

## Resource Subscriptions

Clients can subscribe to resource changes:

```python
from mcp.types import ResourceUpdatedNotification

server = Server("subscribable-resources")

# Track subscriptions
subscriptions: set[str] = set()

@server.subscribe_resource()
async def subscribe(uri: str):
    """Subscribe to resource updates."""
    subscriptions.add(uri)

@server.unsubscribe_resource()
async def unsubscribe(uri: str):
    """Unsubscribe from resource updates."""
    subscriptions.discard(uri)

async def notify_resource_change(uri: str):
    """Notify clients when a resource changes."""
    if uri in subscriptions:
        await server.send_notification(
            ResourceUpdatedNotification(uri=uri)
        )

# Example: File watcher that notifies on changes
async def watch_files():
    async for change in file_watcher(ALLOWED_DIR):
        uri = f"file://{change.path}"
        await notify_resource_change(uri)
```

## Client Usage

### Reading Resources

```python
from mcp import ClientSession

async def use_resources(session: ClientSession):
    # List available resources
    resources = await session.list_resources()
    print(f"Available resources: {len(resources.resources)}")
    
    for resource in resources.resources:
        print(f"  - {resource.uri}: {resource.name}")
    
    # Read a specific resource
    content = await session.read_resource("file://config.json")
    print(f"Config content: {content.contents[0].text}")
```

### Including Resources in LLM Context

```python
async def chat_with_context(user_message: str, session: ClientSession):
    # Read relevant resources for context
    readme = await session.read_resource("docs://readme")
    config = await session.read_resource("config://settings")
    
    # Build context-aware prompt
    system_prompt = f"""You have access to the following project information:

## README
{readme.contents[0].text}

## Configuration
{config.contents[0].text}

Use this context to answer questions about the project."""
    
    # Call LLM with context
    response = await llm.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    return response.choices[0].message.content
```

## Best Practices

```yaml
resource_best_practices:
  uri_design:
    - "Use consistent URI schemes"
    - "Make URIs human-readable"
    - "Include version info if needed (api://v1/users)"
    
  security:
    - "Validate all URIs before processing"
    - "Implement access controls"
    - "Sanitize paths to prevent traversal attacks"
    - "Rate limit resource reads"
    
  performance:
    - "Cache frequently accessed resources"
    - "Paginate large datasets"
    - "Use subscriptions for real-time updates"
    
  content:
    - "Use appropriate MIME types"
    - "Include helpful descriptions"
    - "Provide resource templates for dynamic data"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP Resources Summary                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Resources = READ-ONLY data sources for LLM context             │
│                                                                  │
│  Key Methods:                                                    │
│    • list_resources() - Enumerate available resources           │
│    • read_resource(uri) - Get resource content                  │
│    • list_resource_templates() - Dynamic URI patterns           │
│    • subscribe/unsubscribe - Real-time updates                  │
│                                                                  │
│  Content Types:                                                  │
│    • TextContent - Text, markdown, JSON, code                   │
│    • BlobContent - Images, PDFs, binary data                    │
│                                                                  │
│  Use Cases:                                                      │
│    • File system access                                          │
│    • Database records                                            │
│    • API data                                                    │
│    • Configuration                                               │
│    • Documentation                                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [Tools in MCP](/learn/mcp/mcp-core-concepts/tools) →
