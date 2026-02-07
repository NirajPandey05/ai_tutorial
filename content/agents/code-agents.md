# Code Generation Agents

Learn how to build agents that can write, execute, and debug code autonomously.

## What are Code Agents?

Code agents are AI systems that can understand programming tasks, generate code, execute it, and iterate based on results or errors.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Code Agent Capabilities                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                       CODE AGENT                                 │   │
│   │                                                                  │   │
│   │  Understand Task                                                │   │
│   │       │                                                         │   │
│   │       ▼                                                         │   │
│   │  Generate Code ───────────┐                                     │   │
│   │       │                   │                                     │   │
│   │       ▼                   │                                     │   │
│   │  Execute Code             │                                     │   │
│   │       │                   │                                     │   │
│   │       ▼                   │  Iterate if errors                  │   │
│   │  Analyze Results ─────────┘                                     │   │
│   │       │                                                         │   │
│   │       ▼                                                         │   │
│   │  Return Solution                                                │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   Capabilities:                                                          │
│   ✓ Write code in multiple languages                                    │
│   ✓ Execute code in sandbox                                             │
│   ✓ Debug and fix errors                                                │
│   ✓ Install packages/dependencies                                       │
│   ✓ Read/write files                                                    │
│   ✓ Run tests                                                           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Code Execution Tools

### 1. Python Code Executor

```python
import asyncio
import subprocess
import tempfile
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys

@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    return_value: Any = None
    execution_time: float = 0.0

class PythonExecutor:
    """Execute Python code safely."""
    
    name = "python_executor"
    description = """Execute Python code and return the output.
Use for calculations, data processing, or testing code.
The code runs in an isolated environment."""
    
    def __init__(
        self,
        timeout: int = 30,
        max_output_length: int = 10000
    ):
        self.timeout = timeout
        self.max_output_length = max_output_length
    
    async def run(
        self,
        code: str,
        packages: list[str] = None
    ) -> ExecutionResult:
        """Execute Python code."""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Install packages if needed
            if packages:
                await self._install_packages(packages)
            
            # Run the code
            start_time = asyncio.get_event_loop().time()
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout} seconds"
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            stdout_text = stdout.decode()[:self.max_output_length]
            stderr_text = stderr.decode()[:self.max_output_length]
            
            if process.returncode != 0:
                return ExecutionResult(
                    success=False,
                    output=stdout_text,
                    error=stderr_text,
                    execution_time=execution_time
                )
            
            return ExecutionResult(
                success=True,
                output=stdout_text,
                error=stderr_text if stderr_text else None,
                execution_time=execution_time
            )
            
        finally:
            os.unlink(temp_path)
    
    async def _install_packages(self, packages: list[str]):
        """Install required packages."""
        for package in packages:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pip", "install", package,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
```

### 2. Multi-Language Executor

```python
class CodeExecutor:
    """Execute code in multiple languages."""
    
    LANGUAGE_CONFIGS = {
        "python": {
            "command": [sys.executable],
            "extension": ".py",
            "packages_cmd": [sys.executable, "-m", "pip", "install"]
        },
        "javascript": {
            "command": ["node"],
            "extension": ".js",
            "packages_cmd": ["npm", "install"]
        },
        "bash": {
            "command": ["bash"],
            "extension": ".sh",
            "packages_cmd": None
        }
    }
    
    def __init__(self, timeout: int = 30):
        self.timeout = timeout
    
    async def run(
        self,
        code: str,
        language: str = "python"
    ) -> ExecutionResult:
        """Execute code in the specified language."""
        
        if language not in self.LANGUAGE_CONFIGS:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}"
            )
        
        config = self.LANGUAGE_CONFIGS[language]
        
        # Create temp file with appropriate extension
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=config["extension"],
            delete=False
        ) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            process = await asyncio.create_subprocess_exec(
                *config["command"], temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ExecutionResult(
                    success=False,
                    output="",
                    error="Execution timed out"
                )
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode(),
                error=stderr.decode() if stderr else None
            )
            
        finally:
            os.unlink(temp_path)
```

### 3. Sandboxed Execution with Docker

```python
import docker
from typing import Optional

class DockerExecutor:
    """Execute code in Docker containers for isolation."""
    
    def __init__(self):
        self.client = docker.from_env()
    
    async def run(
        self,
        code: str,
        language: str = "python",
        timeout: int = 30,
        memory_limit: str = "256m"
    ) -> ExecutionResult:
        """Execute code in a Docker container."""
        
        images = {
            "python": "python:3.11-slim",
            "node": "node:18-slim",
        }
        
        if language not in images:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Unsupported language: {language}"
            )
        
        try:
            container = self.client.containers.run(
                images[language],
                command=["python", "-c", code] if language == "python" 
                        else ["node", "-e", code],
                detach=True,
                mem_limit=memory_limit,
                network_disabled=True,  # No network access
                read_only=True,  # Read-only filesystem
            )
            
            # Wait for completion
            result = container.wait(timeout=timeout)
            
            logs = container.logs().decode()
            
            container.remove()
            
            return ExecutionResult(
                success=result["StatusCode"] == 0,
                output=logs,
                error=None if result["StatusCode"] == 0 else logs
            )
            
        except docker.errors.ContainerError as e:
            return ExecutionResult(
                success=False,
                output="",
                error=str(e)
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Docker error: {str(e)}"
            )
```

## Complete Code Agent

```python
from openai import AsyncOpenAI
import json
from typing import List, Dict, Any, Optional

class CodeAgent:
    """Agent that can write, execute, and debug code."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        max_iterations: int = 5
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.executor = PythonExecutor()
        
        self.system_prompt = """You are a coding agent that writes and executes Python code.

When given a task:
1. Think about what code is needed
2. Write clean, well-commented code
3. Execute the code to verify it works
4. If there are errors, analyze and fix them
5. Return the final working solution

Available Tools:
- execute_python(code, packages): Run Python code
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file
- list_files(directory): List files in a directory

Guidelines:
- Always test your code before returning
- Handle errors gracefully
- Use appropriate libraries
- Write clear, readable code
- Include helpful comments"""
    
    async def run(self, task: str) -> dict:
        """Execute a coding task."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {task}"}
        ]
        
        tools = self._get_tools()
        iterations = []
        
        for i in range(self.max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            # No tool calls - we have the final answer
            if not message.tool_calls:
                return {
                    "success": True,
                    "result": message.content,
                    "iterations": iterations
                }
            
            messages.append(message)
            
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                
                result = await self._execute_tool(name, args)
                
                iterations.append({
                    "tool": name,
                    "args": args,
                    "result": result
                })
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
        
        return {
            "success": False,
            "result": "Max iterations reached",
            "iterations": iterations
        }
    
    async def _execute_tool(
        self, 
        name: str, 
        args: Dict[str, Any]
    ) -> dict:
        """Execute a tool and return results."""
        
        if name == "execute_python":
            result = await self.executor.run(
                code=args.get("code", ""),
                packages=args.get("packages", [])
            )
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error
            }
        
        elif name == "read_file":
            try:
                with open(args["path"], "r") as f:
                    content = f.read()
                return {"success": True, "content": content}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif name == "write_file":
            try:
                with open(args["path"], "w") as f:
                    f.write(args["content"])
                return {"success": True, "message": f"Wrote to {args['path']}"}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        elif name == "list_files":
            try:
                files = os.listdir(args.get("directory", "."))
                return {"success": True, "files": files}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        return {"success": False, "error": f"Unknown tool: {name}"}
    
    def _get_tools(self) -> List[dict]:
        """Get tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "execute_python",
                    "description": "Execute Python code and return the output",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute"
                            },
                            "packages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Packages to install before execution"
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files in a directory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "directory": {"type": "string", "default": "."}
                        }
                    }
                }
            }
        ]
```

## Code Agent Patterns

### 1. Test-Driven Development Agent

```python
class TDDCodeAgent(CodeAgent):
    """Agent that writes tests first, then implements."""
    
    async def run_tdd(self, specification: str) -> dict:
        """Implement a feature using TDD."""
        
        # Step 1: Write tests first
        test_task = f"""Write comprehensive unit tests for this specification:

{specification}

Use pytest. Write tests that:
1. Test the main functionality
2. Test edge cases
3. Test error handling

Only write tests, not implementation."""
        
        test_result = await self.run(test_task)
        
        if not test_result["success"]:
            return test_result
        
        # Step 2: Run tests (they should fail)
        test_code = self._extract_code(test_result["result"])
        
        # Step 3: Implement to make tests pass
        impl_task = f"""Implement code to make these tests pass:

{test_code}

Write the implementation code. After writing, run the tests to verify."""
        
        impl_result = await self.run(impl_task)
        
        return {
            "tests": test_result,
            "implementation": impl_result
        }
```

### 2. Debugging Agent

```python
class DebugAgent(CodeAgent):
    """Agent specialized in debugging code."""
    
    async def debug(
        self,
        code: str,
        error: str,
        context: str = ""
    ) -> dict:
        """Debug code that has an error."""
        
        task = f"""Debug this code:

```python
{code}
```

Error encountered:
```
{error}
```

{f"Additional context: {context}" if context else ""}

Steps:
1. Analyze the error message
2. Identify the root cause
3. Fix the code
4. Test the fix
5. Explain what was wrong and how you fixed it"""
        
        return await self.run(task)
```

### 3. Refactoring Agent

```python
class RefactorAgent(CodeAgent):
    """Agent that improves existing code."""
    
    async def refactor(
        self,
        code: str,
        goals: List[str] = None
    ) -> dict:
        """Refactor code for improvement."""
        
        goals = goals or [
            "Improve readability",
            "Follow PEP 8",
            "Add type hints",
            "Improve performance where possible"
        ]
        
        task = f"""Refactor this code:

```python
{code}
```

Goals:
{chr(10).join(f"- {g}" for g in goals)}

Provide:
1. The refactored code
2. Explanation of changes made
3. Any potential issues to be aware of"""
        
        return await self.run(task)
```

## Safety Considerations

```python
class SafeCodeExecutor:
    """Code executor with safety checks."""
    
    DANGEROUS_PATTERNS = [
        "os.system",
        "subprocess",
        "eval(",
        "exec(",
        "__import__",
        "open(",  # Can allow with restrictions
        "requests.",  # Network access
        "urllib",
        "socket",
        "shutil.rmtree",
        "os.remove",
    ]
    
    def is_safe(self, code: str) -> tuple[bool, Optional[str]]:
        """Check if code is safe to execute."""
        
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern in code:
                return False, f"Dangerous pattern detected: {pattern}"
        
        return True, None
    
    async def run_safe(self, code: str) -> ExecutionResult:
        """Run code only if it passes safety checks."""
        
        is_safe, reason = self.is_safe(code)
        
        if not is_safe:
            return ExecutionResult(
                success=False,
                output="",
                error=f"Code rejected: {reason}"
            )
        
        return await self.executor.run(code)
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                 Code Generation Agents - Summary                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Core Components:                                                        │
│    • Code Generator (LLM)                                               │
│    • Code Executor (sandbox)                                            │
│    • File System Access                                                 │
│    • Error Analyzer                                                     │
│                                                                          │
│  Execution Options:                                                      │
│    • Subprocess (basic)                                                 │
│    • Docker containers (isolated)                                       │
│    • Cloud sandboxes (scalable)                                         │
│                                                                          │
│  Agent Patterns:                                                         │
│    • Write-Execute-Fix loop                                             │
│    • Test-Driven Development                                            │
│    • Debug and Repair                                                   │
│    • Refactoring                                                        │
│                                                                          │
│  Safety Measures:                                                        │
│    • Pattern blocking                                                   │
│    • Sandboxed execution                                                │
│    • Resource limits                                                    │
│    • Network isolation                                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Multi-Tool Agents](/learn/agents/agent-capabilities/multi-tool-agents) →
