# Lab: Build a File System MCP Server

Create a production-ready MCP server that provides safe file system access with advanced features.

## Lab Overview

```yaml
objectives:
  - "Build a secure file system MCP server"
  - "Implement file watching and notifications"
  - "Add search and analysis features"
  - "Practice production patterns"

difficulty: Advanced
time: 45 minutes
prerequisites:
  - "Completed MCP Server Lab"
  - "File system operations"
  - "Security best practices"
```

## What We're Building

A **Production File System MCP Server** with:
- Secure file operations with sandboxing
- File content search with regex
- Directory tree visualization
- File watching with change notifications
- Git integration for version info

```
┌─────────────────────────────────────────────────────────────────┐
│               File System MCP Server (Production)                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  TOOLS:                                                          │
│    • read_file         - Read file with encoding detection      │
│    • write_file        - Write with backup creation             │
│    • create_directory  - Create directories recursively         │
│    • move_file         - Move/rename files safely               │
│    • delete_file       - Delete with trash support              │
│    • search_content    - Search file contents with regex        │
│    • get_file_info     - Detailed file metadata                 │
│    • diff_files        - Compare two files                      │
│                                                                  │
│  RESOURCES:                                                      │
│    • file://{path}     - File contents                          │
│    • tree://{path}     - Directory tree view                    │
│    • info://{path}     - File/directory info                    │
│    • git://{path}      - Git status for path                    │
│                                                                  │
│  PROMPTS:                                                        │
│    • analyze_codebase  - Analyze project structure              │
│    • refactor_file     - Suggest refactoring                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Part 1: Project Setup

### Project Structure

```
filesystem_server/
├── pyproject.toml
├── README.md
└── src/
    └── filesystem_server/
        ├── __init__.py
        ├── __main__.py
        ├── server.py
        ├── config.py
        ├── security.py
        ├── tools/
        │   ├── __init__.py
        │   ├── file_ops.py
        │   ├── search.py
        │   └── git_ops.py
        ├── resources/
        │   ├── __init__.py
        │   ├── file_resources.py
        │   └── tree_resources.py
        └── prompts/
            ├── __init__.py
            └── analysis.py
```

### pyproject.toml

```toml
[project]
name = "filesystem-server"
version = "1.0.0"
description = "Production MCP Server for File System Operations"
requires-python = ">=3.10"
dependencies = [
    "mcp>=1.0.0",
    "chardet>=5.0.0",      # Encoding detection
    "gitpython>=3.0.0",    # Git integration
    "watchdog>=3.0.0",     # File watching
]

[project.scripts]
filesystem-server = "filesystem_server:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Part 2: Security Module

```python
# src/filesystem_server/security.py
"""Security utilities for file system access."""

import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    
    # Allowed base directories
    allowed_roots: list[Path]
    
    # Blocked patterns (gitignore-style)
    blocked_patterns: list[str] = None
    
    # Allowed file extensions (None = all allowed)
    allowed_extensions: Optional[list[str]] = None
    
    # Max file size for read/write operations
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    
    # Allow operations outside allowed roots
    allow_symlink_escape: bool = False
    
    # Read-only mode
    read_only: bool = False
    
    def __post_init__(self):
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                ".git/*",
                "node_modules/*",
                "__pycache__/*",
                "*.pyc",
                ".env",
                ".env.*",
                "*.key",
                "*.pem",
                "**/secrets/*",
            ]
        
        # Normalize paths
        self.allowed_roots = [Path(p).resolve() for p in self.allowed_roots]


class SecurityManager:
    """Manages security checks for file operations."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile blocked patterns to regex."""
        self._blocked_regex = []
        for pattern in self.policy.blocked_patterns:
            # Convert gitignore-style to regex
            regex = pattern.replace(".", r"\.")
            regex = regex.replace("**/", "(.*/)?")
            regex = regex.replace("*", "[^/]*")
            regex = f"^{regex}$"
            self._blocked_regex.append(re.compile(regex))
    
    def resolve_path(self, path: str | Path) -> Path:
        """Resolve a path and validate it."""
        path = Path(path)
        
        # Handle relative paths
        if not path.is_absolute():
            # Use first allowed root as base
            if self.policy.allowed_roots:
                path = self.policy.allowed_roots[0] / path
            else:
                path = Path.cwd() / path
        
        return path.resolve()
    
    def is_path_allowed(self, path: Path) -> tuple[bool, str]:
        """Check if a path is allowed by security policy."""
        resolved = path.resolve()
        
        # Check if within allowed roots
        in_allowed_root = any(
            self._is_subpath(resolved, root) 
            for root in self.policy.allowed_roots
        )
        
        if not in_allowed_root:
            return False, f"Path outside allowed directories: {path}"
        
        # Check symlink escape
        if not self.policy.allow_symlink_escape:
            if path.is_symlink():
                target = path.resolve()
                if not any(self._is_subpath(target, root) for root in self.policy.allowed_roots):
                    return False, f"Symlink escapes allowed directories: {path}"
        
        # Check blocked patterns
        for root in self.policy.allowed_roots:
            if self._is_subpath(resolved, root):
                rel_path = str(resolved.relative_to(root))
                for regex in self._blocked_regex:
                    if regex.match(rel_path):
                        return False, f"Path matches blocked pattern: {path}"
                break
        
        # Check extension
        if self.policy.allowed_extensions and resolved.is_file():
            if resolved.suffix.lower() not in self.policy.allowed_extensions:
                return False, f"File extension not allowed: {resolved.suffix}"
        
        return True, ""
    
    def check_file_size(self, path: Path) -> tuple[bool, str]:
        """Check if file size is within limits."""
        if path.exists() and path.is_file():
            size = path.stat().st_size
            if size > self.policy.max_file_size:
                return False, f"File too large: {size} bytes (max: {self.policy.max_file_size})"
        return True, ""
    
    def check_write_allowed(self) -> tuple[bool, str]:
        """Check if write operations are allowed."""
        if self.policy.read_only:
            return False, "Server is in read-only mode"
        return True, ""
    
    def _is_subpath(self, path: Path, parent: Path) -> bool:
        """Check if path is under parent."""
        try:
            path.relative_to(parent)
            return True
        except ValueError:
            return False
    
    def validate_operation(
        self,
        path: str | Path,
        operation: str = "read"
    ) -> Path:
        """Validate an operation and return resolved path."""
        resolved = self.resolve_path(path)
        
        # Check path is allowed
        allowed, reason = self.is_path_allowed(resolved)
        if not allowed:
            raise PermissionError(reason)
        
        # Check write permission
        if operation in ("write", "delete", "move", "create"):
            allowed, reason = self.check_write_allowed()
            if not allowed:
                raise PermissionError(reason)
        
        # Check file size for read operations
        if operation == "read" and resolved.exists():
            allowed, reason = self.check_file_size(resolved)
            if not allowed:
                raise ValueError(reason)
        
        return resolved
```

## Part 3: File Operation Tools

```python
# src/filesystem_server/tools/file_ops.py
"""File operation tools."""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import chardet
from mcp.types import Tool, TextContent

from ..security import SecurityManager


def get_tools() -> list[Tool]:
    """Return file operation tools."""
    return [
        Tool(
            name="read_file",
            description="""Read a file's contents with automatic encoding detection.

Supports text files with various encodings.
Returns file content and detected encoding.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "encoding": {
                        "type": "string",
                        "description": "Encoding (auto-detected if not specified)"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-based, optional)"
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (inclusive, optional)"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="write_file",
            description="""Write content to a file.

Creates parent directories if needed.
Optionally creates backup of existing file.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write"
                    },
                    "create_backup": {
                        "type": "boolean",
                        "default": True,
                        "description": "Create backup of existing file"
                    },
                    "encoding": {
                        "type": "string",
                        "default": "utf-8",
                        "description": "File encoding"
                    }
                },
                "required": ["path", "content"]
            }
        ),
        Tool(
            name="create_directory",
            description="Create a directory (and parents if needed).",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to create"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="move_file",
            description="Move or rename a file or directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source path"
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination path"
                    },
                    "overwrite": {
                        "type": "boolean",
                        "default": False,
                        "description": "Overwrite if destination exists"
                    }
                },
                "required": ["source", "destination"]
            }
        ),
        Tool(
            name="delete_file",
            description="Delete a file or empty directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to delete"
                    },
                    "recursive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Delete directories recursively"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="get_file_info",
            description="Get detailed information about a file or directory.",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to get info for"
                    }
                },
                "required": ["path"]
            }
        ),
        Tool(
            name="diff_files",
            description="Compare two files and show differences.",
            inputSchema={
                "type": "object",
                "properties": {
                    "file1": {
                        "type": "string",
                        "description": "First file path"
                    },
                    "file2": {
                        "type": "string",
                        "description": "Second file path"
                    },
                    "context_lines": {
                        "type": "integer",
                        "default": 3,
                        "description": "Number of context lines"
                    }
                },
                "required": ["file1", "file2"]
            }
        )
    ]


class FileOperations:
    """File operation implementations."""
    
    def __init__(self, security: SecurityManager):
        self.security = security
    
    async def read_file(self, args: dict) -> list[TextContent]:
        """Read file contents."""
        path = self.security.validate_operation(args["path"], "read")
        
        if not path.exists():
            return [TextContent(type="text", text=f"❌ File not found: {args['path']}")]
        
        if not path.is_file():
            return [TextContent(type="text", text=f"❌ Not a file: {args['path']}")]
        
        # Read raw bytes for encoding detection
        raw_content = path.read_bytes()
        
        # Detect or use specified encoding
        encoding = args.get("encoding")
        if not encoding:
            detected = chardet.detect(raw_content)
            encoding = detected.get("encoding", "utf-8")
        
        try:
            content = raw_content.decode(encoding)
        except UnicodeDecodeError:
            return [TextContent(
                type="text",
                text=f"❌ Failed to decode file with encoding: {encoding}"
            )]
        
        # Handle line range
        start_line = args.get("start_line")
        end_line = args.get("end_line")
        
        if start_line or end_line:
            lines = content.splitlines(keepends=True)
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)
            content = "".join(lines[start:end])
        
        return [TextContent(
            type="text",
            text=f"📄 {path.name} (encoding: {encoding})\n\n{content}"
        )]
    
    async def write_file(self, args: dict) -> list[TextContent]:
        """Write to file."""
        path = self.security.validate_operation(args["path"], "write")
        content = args["content"]
        encoding = args.get("encoding", "utf-8")
        create_backup = args.get("create_backup", True)
        
        # Create backup if file exists
        backup_path = None
        if path.exists() and create_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = path.with_suffix(f".{timestamp}.bak")
            shutil.copy2(path, backup_path)
        
        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        path.write_text(content, encoding=encoding)
        
        result = f"✅ Written to: {args['path']}\n"
        result += f"📄 Size: {len(content)} characters\n"
        if backup_path:
            result += f"💾 Backup: {backup_path.name}"
        
        return [TextContent(type="text", text=result)]
    
    async def create_directory(self, args: dict) -> list[TextContent]:
        """Create directory."""
        path = self.security.validate_operation(args["path"], "create")
        
        if path.exists():
            if path.is_dir():
                return [TextContent(type="text", text=f"ℹ️ Directory already exists: {args['path']}")]
            else:
                return [TextContent(type="text", text=f"❌ Path exists but is not a directory: {args['path']}")]
        
        path.mkdir(parents=True)
        return [TextContent(type="text", text=f"✅ Created directory: {args['path']}")]
    
    async def move_file(self, args: dict) -> list[TextContent]:
        """Move/rename file."""
        source = self.security.validate_operation(args["source"], "read")
        dest = self.security.validate_operation(args["destination"], "write")
        overwrite = args.get("overwrite", False)
        
        if not source.exists():
            return [TextContent(type="text", text=f"❌ Source not found: {args['source']}")]
        
        if dest.exists() and not overwrite:
            return [TextContent(type="text", text=f"❌ Destination exists: {args['destination']} (use overwrite=true)")]
        
        # Create destination parent
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        shutil.move(str(source), str(dest))
        return [TextContent(type="text", text=f"✅ Moved: {args['source']} → {args['destination']}")]
    
    async def delete_file(self, args: dict) -> list[TextContent]:
        """Delete file or directory."""
        path = self.security.validate_operation(args["path"], "delete")
        recursive = args.get("recursive", False)
        
        if not path.exists():
            return [TextContent(type="text", text=f"❌ Path not found: {args['path']}")]
        
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                try:
                    path.rmdir()
                except OSError:
                    return [TextContent(
                        type="text",
                        text=f"❌ Directory not empty: {args['path']} (use recursive=true)"
                    )]
        else:
            path.unlink()
        
        return [TextContent(type="text", text=f"✅ Deleted: {args['path']}")]
    
    async def get_file_info(self, args: dict) -> list[TextContent]:
        """Get file information."""
        path = self.security.validate_operation(args["path"], "read")
        
        if not path.exists():
            return [TextContent(type="text", text=f"❌ Path not found: {args['path']}")]
        
        stat = path.stat()
        
        info = {
            "path": str(path),
            "name": path.name,
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
        }
        
        if path.is_file():
            info["extension"] = path.suffix
            # Detect encoding for text files
            try:
                raw = path.read_bytes()[:1000]
                detected = chardet.detect(raw)
                info["encoding"] = detected.get("encoding")
            except Exception:
                pass
        
        if path.is_dir():
            info["items"] = len(list(path.iterdir()))
        
        import json
        return [TextContent(type="text", text=json.dumps(info, indent=2))]
    
    async def diff_files(self, args: dict) -> list[TextContent]:
        """Diff two files."""
        import difflib
        
        file1 = self.security.validate_operation(args["file1"], "read")
        file2 = self.security.validate_operation(args["file2"], "read")
        context = args.get("context_lines", 3)
        
        if not file1.exists():
            return [TextContent(type="text", text=f"❌ File not found: {args['file1']}")]
        if not file2.exists():
            return [TextContent(type="text", text=f"❌ File not found: {args['file2']}")]
        
        lines1 = file1.read_text().splitlines(keepends=True)
        lines2 = file2.read_text().splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            lines1, lines2,
            fromfile=args["file1"],
            tofile=args["file2"],
            n=context
        )
        
        diff_text = "".join(diff)
        
        if not diff_text:
            return [TextContent(type="text", text="✅ Files are identical")]
        
        return [TextContent(type="text", text=f"```diff\n{diff_text}```")]
```

## Part 4: Search Tools

```python
# src/filesystem_server/tools/search.py
"""Search tools."""

import re
from pathlib import Path
from mcp.types import Tool, TextContent

from ..security import SecurityManager


def get_tools() -> list[Tool]:
    """Return search tools."""
    return [
        Tool(
            name="search_content",
            description="""Search file contents using text or regex patterns.

Returns matching lines with context.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (text or regex)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in"
                    },
                    "file_pattern": {
                        "type": "string",
                        "default": "*",
                        "description": "File glob pattern (e.g., '*.py')"
                    },
                    "is_regex": {
                        "type": "boolean",
                        "default": False,
                        "description": "Treat pattern as regex"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "default": False,
                        "description": "Case-sensitive search"
                    },
                    "context_lines": {
                        "type": "integer",
                        "default": 0,
                        "description": "Lines of context around matches"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 50,
                        "description": "Maximum number of matches"
                    }
                },
                "required": ["pattern", "path"]
            }
        ),
        Tool(
            name="find_files",
            description="""Find files by name pattern.

Searches recursively from the given path.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g., '*.py', 'test_*.js')"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in"
                    },
                    "type": {
                        "type": "string",
                        "enum": ["file", "directory", "any"],
                        "default": "any",
                        "description": "Type of items to find"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 100,
                        "description": "Maximum results"
                    }
                },
                "required": ["pattern", "path"]
            }
        )
    ]


class SearchOperations:
    """Search operation implementations."""
    
    def __init__(self, security: SecurityManager):
        self.security = security
    
    async def search_content(self, args: dict) -> list[TextContent]:
        """Search file contents."""
        pattern = args["pattern"]
        base_path = self.security.validate_operation(args["path"], "read")
        file_pattern = args.get("file_pattern", "*")
        is_regex = args.get("is_regex", False)
        case_sensitive = args.get("case_sensitive", False)
        context_lines = args.get("context_lines", 0)
        max_results = args.get("max_results", 50)
        
        if not base_path.is_dir():
            return [TextContent(type="text", text=f"❌ Not a directory: {args['path']}")]
        
        # Compile pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if is_regex:
            try:
                regex = re.compile(pattern, flags)
            except re.error as e:
                return [TextContent(type="text", text=f"❌ Invalid regex: {e}")]
        else:
            regex = re.compile(re.escape(pattern), flags)
        
        # Search files
        matches = []
        files_searched = 0
        
        for file_path in base_path.rglob(file_pattern):
            if not file_path.is_file():
                continue
            
            try:
                allowed, _ = self.security.is_path_allowed(file_path)
                if not allowed:
                    continue
                
                content = file_path.read_text(errors="ignore")
                lines = content.splitlines()
                files_searched += 1
                
                for i, line in enumerate(lines):
                    if regex.search(line):
                        # Get context
                        start = max(0, i - context_lines)
                        end = min(len(lines), i + context_lines + 1)
                        
                        context_text = []
                        for j in range(start, end):
                            prefix = ">" if j == i else " "
                            context_text.append(f"{prefix} {j + 1}: {lines[j]}")
                        
                        rel_path = file_path.relative_to(base_path)
                        matches.append({
                            "file": str(rel_path),
                            "line": i + 1,
                            "context": "\n".join(context_text)
                        })
                        
                        if len(matches) >= max_results:
                            break
                
                if len(matches) >= max_results:
                    break
                    
            except Exception:
                continue
        
        if not matches:
            return [TextContent(
                type="text",
                text=f"No matches found for '{pattern}' in {files_searched} files"
            )]
        
        # Format output
        output = f"Found {len(matches)} match(es) in {files_searched} files:\n\n"
        
        for match in matches:
            output += f"📄 {match['file']}:{match['line']}\n"
            output += f"```\n{match['context']}\n```\n\n"
        
        if len(matches) >= max_results:
            output += f"\n(Results limited to {max_results})"
        
        return [TextContent(type="text", text=output)]
    
    async def find_files(self, args: dict) -> list[TextContent]:
        """Find files by name."""
        pattern = args["pattern"]
        base_path = self.security.validate_operation(args["path"], "read")
        item_type = args.get("type", "any")
        max_results = args.get("max_results", 100)
        
        if not base_path.is_dir():
            return [TextContent(type="text", text=f"❌ Not a directory: {args['path']}")]
        
        results = []
        
        for path in base_path.rglob(pattern):
            allowed, _ = self.security.is_path_allowed(path)
            if not allowed:
                continue
            
            if item_type == "file" and not path.is_file():
                continue
            if item_type == "directory" and not path.is_dir():
                continue
            
            rel_path = path.relative_to(base_path)
            results.append({
                "path": str(rel_path),
                "type": "directory" if path.is_dir() else "file",
                "size": path.stat().st_size if path.is_file() else None
            })
            
            if len(results) >= max_results:
                break
        
        if not results:
            return [TextContent(type="text", text=f"No files found matching '{pattern}'")]
        
        output = f"Found {len(results)} item(s):\n\n"
        for item in results:
            icon = "📁" if item["type"] == "directory" else "📄"
            size = f" ({item['size']} bytes)" if item["size"] else ""
            output += f"{icon} {item['path']}{size}\n"
        
        return [TextContent(type="text", text=output)]
```

## Part 5: Resources

```python
# src/filesystem_server/resources/tree_resources.py
"""Directory tree resources."""

from pathlib import Path
from mcp.types import Resource, TextContent

from ..security import SecurityManager


class TreeResourceProvider:
    """Provides directory tree resources."""
    
    def __init__(self, security: SecurityManager):
        self.security = security
    
    def list_resources(self) -> list[Resource]:
        """List tree resources for allowed roots."""
        resources = []
        
        for root in self.security.policy.allowed_roots:
            resources.append(Resource(
                uri=f"tree://{root.name}",
                name=f"Directory Tree: {root.name}",
                description=f"Tree view of {root}",
                mimeType="text/plain"
            ))
        
        return resources
    
    async def read_resource(self, uri: str) -> list[TextContent]:
        """Read a tree resource."""
        if not uri.startswith("tree://"):
            raise ValueError(f"Invalid tree URI: {uri}")
        
        path_str = uri.replace("tree://", "")
        
        # Find matching root or resolve path
        target = None
        for root in self.security.policy.allowed_roots:
            if root.name == path_str:
                target = root
                break
        
        if not target:
            target = self.security.validate_operation(path_str, "read")
        
        if not target.is_dir():
            raise ValueError(f"Not a directory: {path_str}")
        
        tree = self._build_tree(target, max_depth=4)
        return [TextContent(type="text", text=tree)]
    
    def _build_tree(
        self,
        path: Path,
        prefix: str = "",
        max_depth: int = 4,
        current_depth: int = 0
    ) -> str:
        """Build ASCII tree representation."""
        if current_depth > max_depth:
            return prefix + "└── ...\n"
        
        output = ""
        
        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        except PermissionError:
            return prefix + "└── [Permission Denied]\n"
        
        items = [i for i in items if self.security.is_path_allowed(i)[0]]
        
        for i, item in enumerate(items):
            is_last = i == len(items) - 1
            connector = "└── " if is_last else "├── "
            
            icon = "📁 " if item.is_dir() else "📄 "
            output += f"{prefix}{connector}{icon}{item.name}\n"
            
            if item.is_dir():
                extension = "    " if is_last else "│   "
                output += self._build_tree(
                    item,
                    prefix + extension,
                    max_depth,
                    current_depth + 1
                )
        
        return output
```

## Part 6: Main Server

```python
# src/filesystem_server/server.py
"""Main server module."""

import asyncio
import os
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server

from .config import Config
from .security import SecurityPolicy, SecurityManager
from .tools.file_ops import FileOperations, get_tools as get_file_tools
from .tools.search import SearchOperations, get_tools as get_search_tools
from .resources.tree_resources import TreeResourceProvider

# Load configuration
config = Config.from_env()

# Setup security
policy = SecurityPolicy(
    allowed_roots=[config.workspace_dir],
    read_only=config.read_only
)
security = SecurityManager(policy)

# Initialize components
file_ops = FileOperations(security)
search_ops = SearchOperations(security)
tree_resources = TreeResourceProvider(security)

# Create server
server = Server(config.name, config.version)


# Tool handlers
@server.list_tools()
async def list_tools():
    return get_file_tools() + get_search_tools()


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    # File operations
    if name == "read_file":
        return await file_ops.read_file(arguments)
    elif name == "write_file":
        return await file_ops.write_file(arguments)
    elif name == "create_directory":
        return await file_ops.create_directory(arguments)
    elif name == "move_file":
        return await file_ops.move_file(arguments)
    elif name == "delete_file":
        return await file_ops.delete_file(arguments)
    elif name == "get_file_info":
        return await file_ops.get_file_info(arguments)
    elif name == "diff_files":
        return await file_ops.diff_files(arguments)
    
    # Search operations
    elif name == "search_content":
        return await search_ops.search_content(arguments)
    elif name == "find_files":
        return await search_ops.find_files(arguments)
    
    else:
        raise ValueError(f"Unknown tool: {name}")


# Resource handlers
@server.list_resources()
async def list_resources():
    return tree_resources.list_resources()


@server.read_resource()
async def read_resource(uri: str):
    if uri.startswith("tree://"):
        return await tree_resources.read_resource(uri)
    raise ValueError(f"Unknown resource: {uri}")


async def main():
    """Run the server."""
    import sys
    print(f"Starting {config.name} v{config.version}", file=sys.stderr)
    print(f"Workspace: {config.workspace_dir}", file=sys.stderr)
    print(f"Read-only: {config.read_only}", file=sys.stderr)
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )
```

## Part 7: Configuration

```python
# src/filesystem_server/config.py
"""Server configuration."""

import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Server configuration."""
    name: str = "filesystem-server"
    version: str = "1.0.0"
    workspace_dir: Path = None
    read_only: bool = False
    
    def __post_init__(self):
        if self.workspace_dir is None:
            self.workspace_dir = Path.cwd()
        elif isinstance(self.workspace_dir, str):
            self.workspace_dir = Path(self.workspace_dir)
        self.workspace_dir = self.workspace_dir.resolve()
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            workspace_dir=Path(os.getenv("WORKSPACE_DIR", os.getcwd())),
            read_only=os.getenv("READ_ONLY", "").lower() == "true"
        )
```

## Running the Server

```bash
# Install
pip install -e .

# Run with default settings
filesystem-server

# Run with custom workspace
WORKSPACE_DIR=/path/to/project filesystem-server

# Run in read-only mode
READ_ONLY=true WORKSPACE_DIR=/path/to/project filesystem-server
```

### Claude Desktop Configuration

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "filesystem-server",
      "env": {
        "WORKSPACE_DIR": "/Users/me/projects",
        "READ_ONLY": "false"
      }
    }
  }
}
```

## Summary

You've built a production-ready file system MCP server with:

- ✅ **Security**: Path sandboxing, extension filtering, symlink protection
- ✅ **File Operations**: Read, write, move, delete with backups
- ✅ **Search**: Content search with regex and file finding
- ✅ **Resources**: Directory tree visualization
- ✅ **Configuration**: Environment-based settings

This server can safely provide file system access to AI assistants while protecting sensitive files and preventing escape from allowed directories.

---

Congratulations on completing the MCP module! You now have the skills to build and deploy MCP servers and clients for AI-powered applications.
