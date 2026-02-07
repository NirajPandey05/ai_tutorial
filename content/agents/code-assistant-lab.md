# Lab: Building a Code Assistant Agent

Build a code assistant agent that can write, execute, debug, and explain code.

## Lab Objectives

By the end of this lab, you will:
- Create a code assistant with execution capabilities
- Implement iterative debugging
- Add code explanation features
- Build a conversational coding interface

## Prerequisites

```bash
pip install openai python-dotenv rich pygments
```

## Project Structure

```
code_assistant/
├── agent.py          # Main code assistant
├── executor.py       # Code execution engine
├── analyzer.py       # Code analysis tools
├── main.py           # Entry point
└── workspace/        # Working directory for files
```

## Part 1: Code Executor

### Step 1: Build the Execution Engine (executor.py)

```python
# executor.py
import asyncio
import subprocess
import tempfile
import os
import sys
from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    return_code: int = 0
    execution_time: float = 0.0


class CodeExecutor:
    """Safe code execution engine."""
    
    def __init__(
        self,
        workspace: str = "./workspace",
        timeout: int = 30
    ):
        self.workspace = Path(workspace)
        self.workspace.mkdir(exist_ok=True)
        self.timeout = timeout
    
    async def execute_python(
        self,
        code: str,
        filename: str = None
    ) -> ExecutionResult:
        """Execute Python code."""
        
        # Create file in workspace
        if filename:
            filepath = self.workspace / filename
        else:
            filepath = self.workspace / "temp_script.py"
        
        filepath.write_text(code)
        
        try:
            start_time = asyncio.get_event_loop().time()
            
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(filepath),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace)
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
                    error=f"Execution timed out after {self.timeout}s",
                    return_code=-1
                )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode(),
                error=stderr.decode() if stderr else None,
                return_code=process.returncode,
                execution_time=execution_time
            )
            
        finally:
            # Clean up temp file
            if not filename and filepath.exists():
                filepath.unlink()
    
    async def execute_with_input(
        self,
        code: str,
        input_data: str
    ) -> ExecutionResult:
        """Execute code with stdin input."""
        
        filepath = self.workspace / "temp_script.py"
        filepath.write_text(code)
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(filepath),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=input_data.encode()),
                timeout=self.timeout
            )
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode(),
                error=stderr.decode() if stderr else None,
                return_code=process.returncode
            )
            
        finally:
            if filepath.exists():
                filepath.unlink()
    
    async def run_tests(
        self,
        test_code: str,
        implementation_code: str = None
    ) -> ExecutionResult:
        """Run pytest tests."""
        
        # Write implementation if provided
        if implementation_code:
            impl_path = self.workspace / "implementation.py"
            impl_path.write_text(implementation_code)
        
        # Write test file
        test_path = self.workspace / "test_code.py"
        test_path.write_text(test_code)
        
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "pytest", str(test_path), "-v",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace)
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=60  # Tests may take longer
            )
            
            return ExecutionResult(
                success=process.returncode == 0,
                output=stdout.decode(),
                error=stderr.decode() if stderr else None,
                return_code=process.returncode
            )
            
        finally:
            if test_path.exists():
                test_path.unlink()
    
    def read_file(self, filename: str) -> str:
        """Read a file from workspace."""
        filepath = self.workspace / filename
        if filepath.exists():
            return filepath.read_text()
        raise FileNotFoundError(f"File not found: {filename}")
    
    def write_file(self, filename: str, content: str):
        """Write a file to workspace."""
        filepath = self.workspace / filename
        filepath.write_text(content)
    
    def list_files(self) -> List[str]:
        """List files in workspace."""
        return [f.name for f in self.workspace.iterdir() if f.is_file()]
```

## Part 2: Code Analyzer

### Step 2: Build Analysis Tools (analyzer.py)

```python
# analyzer.py
import ast
import re
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class CodeIssue:
    """An issue found in code."""
    line: int
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: Optional[str] = None


@dataclass
class CodeAnalysis:
    """Result of code analysis."""
    syntax_valid: bool
    issues: List[CodeIssue]
    complexity: int
    functions: List[str]
    imports: List[str]


class CodeAnalyzer:
    """Analyze Python code for issues and structure."""
    
    def analyze(self, code: str) -> CodeAnalysis:
        """Perform comprehensive code analysis."""
        
        issues = []
        functions = []
        imports = []
        syntax_valid = True
        
        # Check syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            syntax_valid = False
            issues.append(CodeIssue(
                line=e.lineno or 1,
                severity="error",
                message=f"Syntax error: {e.msg}",
                suggestion="Fix the syntax error before running"
            ))
            return CodeAnalysis(
                syntax_valid=False,
                issues=issues,
                complexity=0,
                functions=[],
                imports=[]
            )
        
        # Extract functions and imports
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(f"{node.module}")
        
        # Check for common issues
        issues.extend(self._check_common_issues(code))
        
        # Calculate complexity (simplified)
        complexity = self._calculate_complexity(tree)
        
        return CodeAnalysis(
            syntax_valid=syntax_valid,
            issues=issues,
            complexity=complexity,
            functions=functions,
            imports=imports
        )
    
    def _check_common_issues(self, code: str) -> List[CodeIssue]:
        """Check for common code issues."""
        
        issues = []
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for debugging statements
            if 'print(' in line and 'debug' in line.lower():
                issues.append(CodeIssue(
                    line=i,
                    severity="warning",
                    message="Debug print statement found",
                    suggestion="Remove debug statements before production"
                ))
            
            # Check for hardcoded credentials
            if re.search(r'password\s*=\s*["\'][^"\']+["\']', line, re.I):
                issues.append(CodeIssue(
                    line=i,
                    severity="error",
                    message="Hardcoded password detected",
                    suggestion="Use environment variables for credentials"
                ))
            
            # Check for very long lines
            if len(line) > 120:
                issues.append(CodeIssue(
                    line=i,
                    severity="info",
                    message="Line exceeds 120 characters",
                    suggestion="Consider breaking into multiple lines"
                ))
            
            # Check for TODO comments
            if 'TODO' in line or 'FIXME' in line:
                issues.append(CodeIssue(
                    line=i,
                    severity="info",
                    message="TODO/FIXME comment found",
                    suggestion=None
                ))
        
        return issues
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity (simplified)."""
        
        complexity = 1
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def format_analysis(self, analysis: CodeAnalysis) -> str:
        """Format analysis results as string."""
        
        lines = [
            f"Syntax Valid: {'✓' if analysis.syntax_valid else '✗'}",
            f"Complexity Score: {analysis.complexity}",
            f"Functions: {', '.join(analysis.functions) or 'None'}",
            f"Imports: {', '.join(analysis.imports) or 'None'}",
            ""
        ]
        
        if analysis.issues:
            lines.append("Issues Found:")
            for issue in analysis.issues:
                icon = {"error": "❌", "warning": "⚠️", "info": "ℹ️"}[issue.severity]
                lines.append(f"  {icon} Line {issue.line}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"     Suggestion: {issue.suggestion}")
        else:
            lines.append("No issues found ✓")
        
        return "\n".join(lines)
```

## Part 3: Code Assistant Agent

### Step 3: Build the Agent (agent.py)

```python
# agent.py
import json
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

from executor import CodeExecutor, ExecutionResult
from analyzer import CodeAnalyzer, CodeAnalysis


class CodeAssistantAgent:
    """AI-powered code assistant."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4",
        workspace: str = "./workspace"
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.executor = CodeExecutor(workspace)
        self.analyzer = CodeAnalyzer()
        self.console = Console()
        
        self.conversation_history: List[Dict[str, str]] = []
        
        self.system_prompt = """You are an expert Python programming assistant.

You can:
1. Write code to solve problems
2. Execute and test code
3. Debug errors
4. Explain code concepts
5. Refactor and improve code

Available Tools:
- execute_python(code): Run Python code and see output
- analyze_code(code): Analyze code for issues
- read_file(filename): Read a file from workspace
- write_file(filename, content): Write code to a file
- run_tests(test_code): Run pytest tests

Guidelines:
- Always test code before finalizing
- Explain your code with comments
- Handle errors gracefully
- Follow PEP 8 style guidelines
- Write clear, maintainable code"""
    
    async def chat(self, user_message: str) -> str:
        """Process a user message and respond."""
        
        # Add to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            *self.conversation_history
        ]
        
        tools = self._get_tool_definitions()
        
        max_iterations = 10
        for _ in range(max_iterations):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                # Add response to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": message.content
                })
                return message.content
            
            messages.append(message)
            
            # Execute tools
            for tool_call in message.tool_calls:
                result = await self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str)
                })
        
        return "I've reached the maximum number of attempts. Please try rephrasing your request."
    
    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool and display progress."""
        
        if name == "execute_python":
            self.console.print("[dim]Executing code...[/dim]")
            
            code = args.get("code", "")
            self._display_code(code, "python")
            
            result = await self.executor.execute_python(code)
            self._display_execution_result(result)
            
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error,
                "execution_time": result.execution_time
            }
        
        elif name == "analyze_code":
            self.console.print("[dim]Analyzing code...[/dim]")
            
            code = args.get("code", "")
            analysis = self.analyzer.analyze(code)
            analysis_text = self.analyzer.format_analysis(analysis)
            
            self.console.print(Panel(
                analysis_text,
                title="📊 Code Analysis",
                border_style="cyan"
            ))
            
            return {
                "syntax_valid": analysis.syntax_valid,
                "issues": [
                    {"line": i.line, "severity": i.severity, "message": i.message}
                    for i in analysis.issues
                ],
                "complexity": analysis.complexity,
                "functions": analysis.functions
            }
        
        elif name == "read_file":
            filename = args.get("filename", "")
            try:
                content = self.executor.read_file(filename)
                self.console.print(f"[dim]Read file: {filename}[/dim]")
                return {"success": True, "content": content}
            except FileNotFoundError:
                return {"success": False, "error": f"File not found: {filename}"}
        
        elif name == "write_file":
            filename = args.get("filename", "")
            content = args.get("content", "")
            self.executor.write_file(filename, content)
            self.console.print(f"[green]Wrote file: {filename}[/green]")
            return {"success": True, "filename": filename}
        
        elif name == "run_tests":
            self.console.print("[dim]Running tests...[/dim]")
            
            test_code = args.get("test_code", "")
            impl_code = args.get("implementation_code")
            
            result = await self.executor.run_tests(test_code, impl_code)
            
            status = "✓ PASSED" if result.success else "✗ FAILED"
            self.console.print(Panel(
                result.output or result.error or "No output",
                title=f"🧪 Test Results: {status}",
                border_style="green" if result.success else "red"
            ))
            
            return {
                "success": result.success,
                "output": result.output,
                "error": result.error
            }
        
        return {"error": f"Unknown tool: {name}"}
    
    def _display_code(self, code: str, language: str = "python"):
        """Display code with syntax highlighting."""
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        self.console.print(Panel(syntax, title="📝 Code", border_style="blue"))
    
    def _display_execution_result(self, result: ExecutionResult):
        """Display execution result."""
        
        if result.success:
            self.console.print(Panel(
                result.output or "[dim]No output[/dim]",
                title=f"✓ Output (in {result.execution_time:.2f}s)",
                border_style="green"
            ))
        else:
            self.console.print(Panel(
                result.error or "Unknown error",
                title="✗ Error",
                border_style="red"
            ))
    
    def _get_tool_definitions(self) -> List[dict]:
        """Get OpenAI tool definitions."""
        
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
                            }
                        },
                        "required": ["code"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_code",
                    "description": "Analyze Python code for issues, complexity, and structure",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to analyze"
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
                    "description": "Read content of a file from workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of file to read"
                            }
                        },
                        "required": ["filename"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file in workspace",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "description": "Name of file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write"
                            }
                        },
                        "required": ["filename", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_tests",
                    "description": "Run pytest tests",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_code": {
                                "type": "string",
                                "description": "Test code to run"
                            },
                            "implementation_code": {
                                "type": "string",
                                "description": "Implementation code (optional)"
                            }
                        },
                        "required": ["test_code"]
                    }
                }
            }
        ]
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
```

## Part 4: Main Application

### Step 4: Create Entry Point (main.py)

```python
# main.py
import asyncio
import os
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

from agent import CodeAssistantAgent


async def main():
    """Run the code assistant."""
    
    load_dotenv()
    console = Console()
    
    console.print(Panel(
        "[bold cyan]🤖 Code Assistant[/bold cyan]\n"
        "Your AI-powered programming helper.\n\n"
        "Commands:\n"
        "  [dim]/clear[/dim] - Clear conversation\n"
        "  [dim]/files[/dim] - List workspace files\n"
        "  [dim]/quit[/dim]  - Exit",
        title="Welcome"
    ))
    
    agent = CodeAssistantAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    while True:
        console.print()
        user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
        
        # Handle commands
        if user_input.lower() == "/quit":
            console.print("[yellow]Goodbye![/yellow]")
            break
        
        if user_input.lower() == "/clear":
            agent.clear_history()
            console.print("[dim]Conversation cleared[/dim]")
            continue
        
        if user_input.lower() == "/files":
            files = agent.executor.list_files()
            console.print(f"[dim]Workspace files: {', '.join(files) or 'None'}[/dim]")
            continue
        
        if not user_input.strip():
            continue
        
        try:
            response = await agent.chat(user_input)
            console.print()
            console.print(Panel(
                Markdown(response),
                title="🤖 Assistant",
                border_style="green"
            ))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def demo():
    """Run a demonstration."""
    
    load_dotenv()
    console = Console()
    
    agent = CodeAssistantAgent(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4"
    )
    
    demo_tasks = [
        "Write a function to check if a number is prime",
        "Now add some test cases for the prime function",
        "Can you optimize the function for large numbers?",
    ]
    
    for task in demo_tasks:
        console.print(f"\n[bold cyan]User:[/bold cyan] {task}")
        response = await agent.chat(task)
        console.print(Panel(
            Markdown(response),
            title="🤖 Assistant",
            border_style="green"
        ))
        console.print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        asyncio.run(demo())
    else:
        asyncio.run(main())
```

## Running the Lab

```bash
# Interactive mode
python main.py

# Demo mode
python main.py demo
```

## Example Interactions

```
You: Write a function to calculate fibonacci numbers efficiently

🤖 Assistant:
I'll write an efficient fibonacci function using memoization:

📝 Code:
def fibonacci(n: int, memo: dict = {}) -> int:
    """Calculate nth fibonacci number with memoization."""
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)
    return memo[n]

# Test
for i in range(10):
    print(f"F({i}) = {fibonacci(i)}")

✓ Output (in 0.05s):
F(0) = 0
F(1) = 1
F(2) = 1
...

The function uses memoization to avoid recalculating values,
making it O(n) time complexity instead of O(2^n).
```

## Exercises

1. **Add Language Support**: Extend to support JavaScript execution
2. **Add Debugging Mode**: Step through code execution
3. **Add Code Formatting**: Auto-format code with black
4. **Add Git Integration**: Track code changes

## Summary

You've built a code assistant that can:
- ✅ Write code based on descriptions
- ✅ Execute code safely
- ✅ Analyze code for issues
- ✅ Debug and fix errors
- ✅ Maintain conversation context
- ✅ Read/write files

Congratulations on completing the AI Agents module! 🎉
