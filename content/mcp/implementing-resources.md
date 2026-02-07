# Implementing Resources in MCP Servers

Learn how to expose data sources as resources in your MCP server, enabling LLMs to access contextual information.

## Resource Implementation Fundamentals

Resources provide read-only access to data that can be included in LLM context. They're addressed by URIs and can represent files, database records, API responses, or any other data.

```yaml
resource_design_principles:
  addressability: "Every resource has a unique URI"
  read_only: "Resources provide data, not actions"
  typed: "Resources have MIME types for proper handling"
  describable: "Resources have names and descriptions"
```

## Basic Resource Implementation

### Step 1: Define Resource Structure

```python
from mcp.server import Server
from mcp.types import Resource, TextContent

server = Server("resource-server")

# Static resource list
AVAILABLE_RESOURCES = [
    Resource(
        uri="config://app-settings",
        name="Application Settings",
        description="Current application configuration",
        mimeType="application/json"
    ),
    Resource(
        uri="docs://readme",
        name="README",
        description="Project documentation",
        mimeType="text/markdown"
    ),
    Resource(
        uri="logs://recent",
        name="Recent Logs",
        description="Last 100 log entries",
        mimeType="text/plain"
    )
]

@server.list_resources()
async def list_resources():
    """Return available resources."""
    return AVAILABLE_RESOURCES
```

### Step 2: Implement Resource Reading

```python
import json

# Resource data (in practice, this would come from files, databases, etc.)
RESOURCE_DATA = {
    "config://app-settings": {
        "debug": True,
        "version": "1.0.0",
        "features": {
            "auth": True,
            "caching": True
        }
    },
    "docs://readme": """# My Project

Welcome to my project!

## Features
- Feature A
- Feature B

## Getting Started
Run `python main.py` to start.
""",
    "logs://recent": """2025-01-21 10:00:00 INFO Starting application
2025-01-21 10:00:01 INFO Database connected
2025-01-21 10:00:02 INFO Server listening on port 8080
"""
}

@server.read_resource()
async def read_resource(uri: str):
    """Read a specific resource by URI."""
    
    if uri not in RESOURCE_DATA:
        raise ValueError(f"Resource not found: {uri}")
    
    data = RESOURCE_DATA[uri]
    
    # Format based on data type
    if isinstance(data, dict):
        text = json.dumps(data, indent=2)
    else:
        text = str(data)
    
    return [TextContent(type="text", text=text)]
```

## File System Resources

Expose file system contents as resources:

```python
import os
from pathlib import Path
import mimetypes

class FileSystemResourceProvider:
    """Provide file system access as MCP resources."""
    
    def __init__(self, base_path: str, allowed_extensions: list[str] = None):
        self.base_path = Path(base_path).resolve()
        self.allowed_extensions = allowed_extensions or [
            ".txt", ".md", ".json", ".yaml", ".yml",
            ".py", ".js", ".ts", ".html", ".css"
        ]
    
    def _is_allowed(self, path: Path) -> bool:
        """Check if file is allowed."""
        # Must be within base path
        try:
            path.resolve().relative_to(self.base_path)
        except ValueError:
            return False
        
        # Must have allowed extension
        if self.allowed_extensions:
            if path.suffix.lower() not in self.allowed_extensions:
                return False
        
        return True
    
    def _get_mime_type(self, path: Path) -> str:
        """Get MIME type for a file."""
        mime_type, _ = mimetypes.guess_type(str(path))
        return mime_type or "application/octet-stream"
    
    def list_resources(self) -> list[Resource]:
        """List all files as resources."""
        resources = []
        
        for file_path in self.base_path.rglob("*"):
            if file_path.is_file() and self._is_allowed(file_path):
                rel_path = file_path.relative_to(self.base_path)
                
                resources.append(Resource(
                    uri=f"file://{rel_path}",
                    name=file_path.name,
                    description=f"File: {rel_path}",
                    mimeType=self._get_mime_type(file_path)
                ))
        
        return resources
    
    async def read_resource(self, uri: str) -> list[TextContent]:
        """Read a file resource."""
        if not uri.startswith("file://"):
            raise ValueError(f"Invalid file URI: {uri}")
        
        rel_path = uri[7:]  # Remove "file://"
        file_path = self.base_path / rel_path
        
        # Security check
        if not self._is_allowed(file_path):
            raise ValueError(f"Access denied: {rel_path}")
        
        if not file_path.exists():
            raise ValueError(f"File not found: {rel_path}")
        
        # Read content
        content = file_path.read_text(encoding="utf-8")
        
        return [TextContent(type="text", text=content)]


# Usage
file_provider = FileSystemResourceProvider(
    base_path="./project",
    allowed_extensions=[".py", ".md", ".json"]
)

@server.list_resources()
async def list_resources():
    return file_provider.list_resources()

@server.read_resource()
async def read_resource(uri: str):
    return await file_provider.read_resource(uri)
```

## Database Resources

Expose database tables and queries as resources:

```python
import sqlite3
import json
from contextlib import contextmanager

class DatabaseResourceProvider:
    """Provide database access as MCP resources."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def list_resources(self) -> list[Resource]:
        """List database tables as resources."""
        resources = []
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            
            for (table_name,) in cursor.fetchall():
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                
                # Get column info
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row["name"] for row in cursor.fetchall()]
                
                resources.append(Resource(
                    uri=f"db://table/{table_name}",
                    name=f"Table: {table_name}",
                    description=f"Database table with {count} rows. "
                               f"Columns: {', '.join(columns)}",
                    mimeType="application/json"
                ))
                
                # Also add a schema resource
                resources.append(Resource(
                    uri=f"db://schema/{table_name}",
                    name=f"Schema: {table_name}",
                    description=f"Schema definition for {table_name}",
                    mimeType="application/json"
                ))
        
        return resources
    
    async def read_resource(self, uri: str) -> list[TextContent]:
        """Read a database resource."""
        
        if uri.startswith("db://table/"):
            table_name = uri.replace("db://table/", "")
            return await self._read_table(table_name)
        
        elif uri.startswith("db://schema/"):
            table_name = uri.replace("db://schema/", "")
            return await self._read_schema(table_name)
        
        raise ValueError(f"Unknown database resource: {uri}")
    
    async def _read_table(self, table_name: str) -> list[TextContent]:
        """Read table contents."""
        # Validate table name (prevent SQL injection)
        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 100")
            
            rows = [dict(row) for row in cursor.fetchall()]
            
            return [TextContent(
                type="text",
                text=json.dumps(rows, indent=2, default=str)
            )]
    
    async def _read_schema(self, table_name: str) -> list[TextContent]:
        """Read table schema."""
        if not table_name.isidentifier():
            raise ValueError(f"Invalid table name: {table_name}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            
            columns = []
            for row in cursor.fetchall():
                columns.append({
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not row["notnull"],
                    "primary_key": bool(row["pk"]),
                    "default": row["dflt_value"]
                })
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "table": table_name,
                    "columns": columns
                }, indent=2)
            )]


# Usage
db_provider = DatabaseResourceProvider("./app.db")

@server.list_resources()
async def list_resources():
    return db_provider.list_resources()

@server.read_resource()
async def read_resource(uri: str):
    return await db_provider.read_resource(uri)
```

## API Resources

Expose external API data as resources:

```python
import httpx
from datetime import datetime, timedelta
from typing import Optional

class CachedAPIResourceProvider:
    """Provide API data as MCP resources with caching."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self._cache: dict[str, tuple[datetime, str]] = {}
        self._cache_ttl = timedelta(minutes=5)
    
    def _is_cached(self, uri: str) -> bool:
        if uri not in self._cache:
            return False
        cached_time, _ = self._cache[uri]
        return datetime.now() - cached_time < self._cache_ttl
    
    def _get_cached(self, uri: str) -> Optional[str]:
        if self._is_cached(uri):
            return self._cache[uri][1]
        return None
    
    def _set_cached(self, uri: str, data: str):
        self._cache[uri] = (datetime.now(), data)
    
    def list_resources(self) -> list[Resource]:
        """Define available API resources."""
        return [
            Resource(
                uri="api://users",
                name="All Users",
                description="List of all users from the API",
                mimeType="application/json"
            ),
            Resource(
                uri="api://stats",
                name="Statistics",
                description="Current system statistics",
                mimeType="application/json"
            ),
            Resource(
                uri="api://config",
                name="Remote Config",
                description="Remote configuration settings",
                mimeType="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> list[TextContent]:
        """Read an API resource."""
        
        # Check cache first
        cached = self._get_cached(uri)
        if cached:
            return [TextContent(type="text", text=cached)]
        
        # Map URI to API endpoint
        endpoint_map = {
            "api://users": "/users",
            "api://stats": "/stats",
            "api://config": "/config"
        }
        
        if uri not in endpoint_map:
            raise ValueError(f"Unknown API resource: {uri}")
        
        endpoint = endpoint_map[uri]
        
        # Fetch from API
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        async with httpx.AsyncClient(base_url=self.base_url) as client:
            try:
                response = await client.get(endpoint, headers=headers)
                response.raise_for_status()
                
                data = json.dumps(response.json(), indent=2)
                
                # Cache the result
                self._set_cached(uri, data)
                
                return [TextContent(type="text", text=data)]
                
            except httpx.HTTPError as e:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": f"API request failed: {str(e)}",
                        "uri": uri
                    })
                )]
```

## Resource Templates

Support dynamic resources with URI templates:

```python
from mcp.types import ResourceTemplate

class TemplatedResourceProvider:
    """Resources with dynamic URI templates."""
    
    def list_resource_templates(self) -> list[ResourceTemplate]:
        """Define resource templates for dynamic URIs."""
        return [
            ResourceTemplate(
                uriTemplate="user://{user_id}",
                name="User Profile",
                description="Get profile for a specific user by ID",
                mimeType="application/json"
            ),
            ResourceTemplate(
                uriTemplate="user://{user_id}/posts",
                name="User Posts",
                description="Get posts for a specific user",
                mimeType="application/json"
            ),
            ResourceTemplate(
                uriTemplate="date://{year}/{month}/{day}",
                name="Daily Report",
                description="Get report for a specific date",
                mimeType="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> list[TextContent]:
        """Read a templated resource."""
        import re
        
        # Match user profile
        match = re.match(r"user://(\d+)$", uri)
        if match:
            user_id = match.group(1)
            return await self._get_user_profile(user_id)
        
        # Match user posts
        match = re.match(r"user://(\d+)/posts$", uri)
        if match:
            user_id = match.group(1)
            return await self._get_user_posts(user_id)
        
        # Match daily report
        match = re.match(r"date://(\d{4})/(\d{2})/(\d{2})$", uri)
        if match:
            year, month, day = match.groups()
            return await self._get_daily_report(year, month, day)
        
        raise ValueError(f"Cannot match resource URI: {uri}")
    
    async def _get_user_profile(self, user_id: str) -> list[TextContent]:
        # Fetch user data...
        user = {"id": user_id, "name": f"User {user_id}", "email": f"user{user_id}@example.com"}
        return [TextContent(type="text", text=json.dumps(user, indent=2))]
    
    async def _get_user_posts(self, user_id: str) -> list[TextContent]:
        # Fetch user posts...
        posts = [
            {"id": 1, "user_id": user_id, "title": "First Post"},
            {"id": 2, "user_id": user_id, "title": "Second Post"}
        ]
        return [TextContent(type="text", text=json.dumps(posts, indent=2))]
    
    async def _get_daily_report(self, year: str, month: str, day: str) -> list[TextContent]:
        # Generate report...
        report = {
            "date": f"{year}-{month}-{day}",
            "metrics": {"visits": 1000, "signups": 50}
        }
        return [TextContent(type="text", text=json.dumps(report, indent=2))]


# Register with server
template_provider = TemplatedResourceProvider()

@server.list_resource_templates()
async def list_resource_templates():
    return template_provider.list_resource_templates()
```

## Resource Subscriptions

Allow clients to subscribe to resource changes:

```python
import asyncio
from typing import Set

class SubscribableResourceProvider:
    """Resource provider with subscription support."""
    
    def __init__(self):
        self._subscriptions: Set[str] = set()
        self._data: dict[str, str] = {
            "live://status": '{"status": "online"}',
            "live://metrics": '{"cpu": 45, "memory": 62}'
        }
    
    def list_resources(self) -> list[Resource]:
        return [
            Resource(
                uri="live://status",
                name="Live Status",
                description="Real-time system status (subscribable)",
                mimeType="application/json"
            ),
            Resource(
                uri="live://metrics",
                name="Live Metrics",
                description="Real-time system metrics (subscribable)",
                mimeType="application/json"
            )
        ]
    
    async def read_resource(self, uri: str) -> list[TextContent]:
        if uri not in self._data:
            raise ValueError(f"Unknown resource: {uri}")
        
        return [TextContent(type="text", text=self._data[uri])]
    
    async def subscribe(self, uri: str):
        """Subscribe to resource updates."""
        self._subscriptions.add(uri)
    
    async def unsubscribe(self, uri: str):
        """Unsubscribe from resource updates."""
        self._subscriptions.discard(uri)
    
    async def update_resource(self, uri: str, new_data: str):
        """Update a resource and notify subscribers."""
        self._data[uri] = new_data
        
        if uri in self._subscriptions:
            # Server would send notification here
            await self._notify_change(uri)
    
    async def _notify_change(self, uri: str):
        """Notify clients of resource change."""
        # This would trigger a notification to subscribed clients
        # The actual implementation depends on your server setup
        pass


# Register subscription handlers
sub_provider = SubscribableResourceProvider()

@server.subscribe_resource()
async def subscribe_resource(uri: str):
    await sub_provider.subscribe(uri)

@server.unsubscribe_resource()
async def unsubscribe_resource(uri: str):
    await sub_provider.unsubscribe(uri)
```

## Binary Resources

Handle binary data like images:

```python
import base64
from mcp.types import BlobContent

class BinaryResourceProvider:
    """Provide binary resources (images, PDFs, etc.)."""
    
    def __init__(self, assets_dir: str):
        self.assets_dir = Path(assets_dir)
    
    def list_resources(self) -> list[Resource]:
        resources = []
        
        binary_extensions = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip"}
        
        for file_path in self.assets_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in binary_extensions:
                rel_path = file_path.relative_to(self.assets_dir)
                mime_type, _ = mimetypes.guess_type(str(file_path))
                
                resources.append(Resource(
                    uri=f"asset://{rel_path}",
                    name=file_path.name,
                    description=f"Binary asset: {rel_path}",
                    mimeType=mime_type or "application/octet-stream"
                ))
        
        return resources
    
    async def read_resource(self, uri: str) -> list:
        """Read a binary resource."""
        if not uri.startswith("asset://"):
            raise ValueError(f"Invalid asset URI: {uri}")
        
        rel_path = uri[8:]  # Remove "asset://"
        file_path = self.assets_dir / rel_path
        
        # Security check
        if not file_path.resolve().is_relative_to(self.assets_dir.resolve()):
            raise ValueError("Access denied")
        
        if not file_path.exists():
            raise ValueError(f"Asset not found: {rel_path}")
        
        # Read binary content
        content = file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        return [BlobContent(
            type="blob",
            data=base64.b64encode(content).decode("utf-8"),
            mimeType=mime_type or "application/octet-stream"
        )]
```

## Composite Resource Provider

Combine multiple providers:

```python
class CompositeResourceProvider:
    """Combine multiple resource providers."""
    
    def __init__(self):
        self._providers: list[tuple[str, Any]] = []
    
    def add_provider(self, prefix: str, provider):
        """Add a provider for a URI prefix."""
        self._providers.append((prefix, provider))
    
    def list_resources(self) -> list[Resource]:
        """List resources from all providers."""
        resources = []
        for prefix, provider in self._providers:
            resources.extend(provider.list_resources())
        return resources
    
    async def read_resource(self, uri: str) -> list:
        """Route to appropriate provider."""
        for prefix, provider in self._providers:
            if uri.startswith(prefix):
                return await provider.read_resource(uri)
        
        raise ValueError(f"No provider for URI: {uri}")


# Usage
composite = CompositeResourceProvider()
composite.add_provider("file://", FileSystemResourceProvider("./files"))
composite.add_provider("db://", DatabaseResourceProvider("./app.db"))
composite.add_provider("api://", APIResourceProvider("https://api.example.com"))
composite.add_provider("asset://", BinaryResourceProvider("./assets"))

@server.list_resources()
async def list_resources():
    return composite.list_resources()

@server.read_resource()
async def read_resource(uri: str):
    return await composite.read_resource(uri)
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│             Implementing Resources - Summary                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Resource Types:                                                 │
│    • File System: Local files as resources                      │
│    • Database: Tables and queries as resources                  │
│    • API: External API data with caching                        │
│    • Templates: Dynamic URIs with parameters                    │
│    • Binary: Images, PDFs, and other binary data                │
│                                                                  │
│  Key Patterns:                                                   │
│    • Provider classes for different data sources                │
│    • URI-based routing and parsing                              │
│    • Caching for expensive operations                           │
│    • Templates for dynamic resources                            │
│    • Subscriptions for real-time updates                        │
│                                                                  │
│  Best Practices:                                                 │
│    • Validate and sanitize all URIs                             │
│    • Use appropriate MIME types                                 │
│    • Implement caching where beneficial                         │
│    • Handle errors gracefully                                   │
│    • Document resource schemas                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [Build an MCP Server Lab](/learn/mcp/mcp-servers/server-lab) →
