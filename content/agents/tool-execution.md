# Tool Execution & Error Handling

Learn how to safely execute tools, handle errors, and manage complex execution scenarios.

## The Tool Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Tool Execution Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   LLM Output          Your Code              Result                     │
│   ──────────          ─────────              ──────                     │
│                                                                          │
│   ┌──────────┐    ┌──────────────┐    ┌──────────────┐                 │
│   │  Parse   │───►│   Validate   │───►│   Execute    │                 │
│   │  Request │    │   Arguments  │    │   Function   │                 │
│   └──────────┘    └──────────────┘    └──────────────┘                 │
│                          │                    │                         │
│                          ▼                    ▼                         │
│                   ┌──────────────┐    ┌──────────────┐                 │
│                   │Handle Invalid│    │ Handle Errors│                 │
│                   │  Arguments   │    │ & Timeouts   │                 │
│                   └──────────────┘    └──────────────┘                 │
│                                              │                          │
│                                              ▼                          │
│                                       ┌──────────────┐                 │
│                                       │   Format     │                 │
│                                       │   Response   │                 │
│                                       └──────────────┘                 │
│                                              │                          │
│                                              ▼                          │
│                                       Return to LLM                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Building a Robust Tool Executor

```python
import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    INVALID_ARGS = "invalid_arguments"
    UNKNOWN_TOOL = "unknown_tool"


@dataclass
class ExecutionResult:
    """Result of tool execution."""
    status: ExecutionStatus
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_llm_response(self) -> str:
        """Format result for sending back to LLM."""
        if self.status == ExecutionStatus.SUCCESS:
            return self.output or "Success (no output)"
        else:
            return f"Error ({self.status.value}): {self.error}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "execution_time": self.execution_time,
            "metadata": self.metadata
        }


@dataclass
class ToolConfig:
    """Configuration for a tool."""
    name: str
    function: Callable
    schema: Dict[str, Any]
    timeout: int = 30
    requires_confirmation: bool = False
    max_retries: int = 1
    rate_limit: Optional[int] = None  # calls per minute


class ToolExecutor:
    """Robust tool executor with error handling and validation."""
    
    def __init__(self):
        self.tools: Dict[str, ToolConfig] = {}
        self.execution_history: list = []
        self.rate_limit_counters: Dict[str, list] = {}
    
    def register_tool(self, config: ToolConfig):
        """Register a tool with its configuration."""
        self.tools[config.name] = config
        logger.info(f"Registered tool: {config.name}")
    
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> ExecutionResult:
        """Execute a tool with full error handling."""
        
        start_time = datetime.now()
        
        # Check if tool exists
        if tool_name not in self.tools:
            return ExecutionResult(
                status=ExecutionStatus.UNKNOWN_TOOL,
                error=f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
            )
        
        config = self.tools[tool_name]
        
        # Check rate limit
        if not self._check_rate_limit(tool_name, config):
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"Rate limit exceeded for {tool_name}"
            )
        
        # Validate arguments
        validation_error = self._validate_arguments(config, arguments)
        if validation_error:
            return ExecutionResult(
                status=ExecutionStatus.INVALID_ARGS,
                error=validation_error
            )
        
        # Execute with retry logic
        result = await self._execute_with_retry(config, arguments, context)
        
        # Calculate execution time
        result.execution_time = (datetime.now() - start_time).total_seconds()
        
        # Log execution
        self._log_execution(tool_name, arguments, result)
        
        return result
    
    def _validate_arguments(
        self, 
        config: ToolConfig, 
        arguments: Dict[str, Any]
    ) -> Optional[str]:
        """Validate arguments against the tool's schema."""
        
        schema = config.schema.get("function", {}).get("parameters", {})
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        
        # Check required parameters
        for param in required:
            if param not in arguments:
                return f"Missing required parameter: {param}"
        
        # Validate types
        for param, value in arguments.items():
            if param not in properties:
                continue  # Allow extra params
            
            prop_schema = properties[param]
            expected_type = prop_schema.get("type")
            
            # Type checking
            type_map = {
                "string": str,
                "integer": int,
                "number": (int, float),
                "boolean": bool,
                "array": list,
                "object": dict
            }
            
            if expected_type in type_map:
                if not isinstance(value, type_map[expected_type]):
                    return f"Parameter '{param}' should be {expected_type}, got {type(value).__name__}"
            
            # Enum validation
            if "enum" in prop_schema:
                if value not in prop_schema["enum"]:
                    return f"Parameter '{param}' must be one of {prop_schema['enum']}"
            
            # Range validation
            if expected_type in ("integer", "number"):
                if "minimum" in prop_schema and value < prop_schema["minimum"]:
                    return f"Parameter '{param}' must be >= {prop_schema['minimum']}"
                if "maximum" in prop_schema and value > prop_schema["maximum"]:
                    return f"Parameter '{param}' must be <= {prop_schema['maximum']}"
        
        return None  # Validation passed
    
    async def _execute_with_retry(
        self,
        config: ToolConfig,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Execute tool with retry logic."""
        
        last_error = None
        
        for attempt in range(config.max_retries):
            try:
                result = await self._execute_single(config, arguments, context)
                if result.status == ExecutionStatus.SUCCESS:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < config.max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))  # Exponential backoff
        
        return ExecutionResult(
            status=ExecutionStatus.ERROR,
            error=f"Failed after {config.max_retries} attempts: {last_error}"
        )
    
    async def _execute_single(
        self,
        config: ToolConfig,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> ExecutionResult:
        """Execute a single attempt of a tool."""
        
        try:
            # Add context if function accepts it
            if context:
                arguments = {**arguments, "_context": context}
            
            # Execute with timeout
            if asyncio.iscoroutinefunction(config.function):
                result = await asyncio.wait_for(
                    config.function(**arguments),
                    timeout=config.timeout
                )
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, lambda: config.function(**arguments)),
                    timeout=config.timeout
                )
            
            # Format result
            if isinstance(result, str):
                output = result
            elif isinstance(result, dict):
                output = json.dumps(result, indent=2)
            else:
                output = str(result)
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=output
            )
            
        except asyncio.TimeoutError:
            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                error=f"Tool execution timed out after {config.timeout} seconds"
            )
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error=f"{type(e).__name__}: {str(e)}"
            )
    
    def _check_rate_limit(self, tool_name: str, config: ToolConfig) -> bool:
        """Check if tool is within rate limits."""
        
        if not config.rate_limit:
            return True
        
        now = datetime.now()
        minute_ago = now.timestamp() - 60
        
        if tool_name not in self.rate_limit_counters:
            self.rate_limit_counters[tool_name] = []
        
        # Remove old entries
        self.rate_limit_counters[tool_name] = [
            t for t in self.rate_limit_counters[tool_name]
            if t > minute_ago
        ]
        
        # Check limit
        if len(self.rate_limit_counters[tool_name]) >= config.rate_limit:
            return False
        
        # Record this call
        self.rate_limit_counters[tool_name].append(now.timestamp())
        return True
    
    def _log_execution(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        result: ExecutionResult
    ):
        """Log tool execution for monitoring."""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "arguments": arguments,
            "result": result.to_dict()
        }
        
        self.execution_history.append(log_entry)
        
        # Log to file/monitoring system
        if result.status == ExecutionStatus.SUCCESS:
            logger.info(f"Tool {tool_name} executed successfully in {result.execution_time:.2f}s")
        else:
            logger.error(f"Tool {tool_name} failed: {result.error}")
```

## Error Handling Strategies

### 1. Graceful Degradation

```python
class GracefulToolExecutor:
    """Executor that provides fallback responses."""
    
    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.fallbacks: Dict[str, Callable] = {}
    
    def register_fallback(self, tool_name: str, fallback: Callable):
        """Register a fallback for when a tool fails."""
        self.fallbacks[tool_name] = fallback
    
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> ExecutionResult:
        """Execute with fallback support."""
        
        result = await self.executor.execute(tool_name, arguments)
        
        if result.status != ExecutionStatus.SUCCESS:
            if tool_name in self.fallbacks:
                try:
                    fallback_result = self.fallbacks[tool_name](arguments)
                    return ExecutionResult(
                        status=ExecutionStatus.SUCCESS,
                        output=fallback_result,
                        metadata={"fallback": True}
                    )
                except Exception as e:
                    logger.error(f"Fallback also failed: {e}")
        
        return result


# Example: Web search with fallback
executor = GracefulToolExecutor(ToolExecutor())

def search_fallback(args: Dict) -> str:
    """Fallback when search fails."""
    query = args.get("query", "")
    return f"""I wasn't able to search for "{query}" due to a technical issue.
However, I can try to answer based on my training data.
What would you like to know?"""

executor.register_fallback("web_search", search_fallback)
```

### 2. Error Recovery Prompts

```python
def format_error_for_llm(result: ExecutionResult, tool_name: str) -> str:
    """Format errors to help LLM recover."""
    
    error_templates = {
        ExecutionStatus.TIMEOUT: """
The {tool_name} tool timed out. This might mean:
- The query was too complex
- External service is slow

You can try:
1. Breaking the request into smaller parts
2. Using a different approach
3. Asking the user to try again later
""",
        
        ExecutionStatus.INVALID_ARGS: """
The {tool_name} tool received invalid arguments: {error}

Please check:
1. Required parameters are provided
2. Parameter types are correct
3. Values are within allowed ranges

Try calling the tool again with corrected arguments.
""",
        
        ExecutionStatus.ERROR: """
The {tool_name} tool encountered an error: {error}

This might be a temporary issue. You can:
1. Retry the operation
2. Try an alternative approach
3. Inform the user about the limitation
""",

        ExecutionStatus.UNKNOWN_TOOL: """
The tool '{tool_name}' doesn't exist. Available tools are listed in your instructions.
Please use one of the available tools or inform the user that this action isn't supported.
"""
    }
    
    template = error_templates.get(result.status, "An error occurred: {error}")
    return template.format(
        tool_name=tool_name,
        error=result.error
    )
```

### 3. Human-in-the-Loop for Risky Operations

```python
class ConfirmationRequired(Exception):
    """Exception raised when human confirmation is needed."""
    pass


class SafeToolExecutor:
    """Executor that requires confirmation for risky operations."""
    
    def __init__(self, executor: ToolExecutor):
        self.executor = executor
        self.pending_confirmations: Dict[str, Dict] = {}
    
    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        confirmed: bool = False
    ) -> Union[ExecutionResult, Dict]:
        """Execute with confirmation for risky tools."""
        
        config = self.executor.tools.get(tool_name)
        
        if config and config.requires_confirmation and not confirmed:
            # Generate confirmation request
            confirmation_id = f"confirm_{hash(json.dumps(arguments))}"
            
            self.pending_confirmations[confirmation_id] = {
                "tool_name": tool_name,
                "arguments": arguments,
                "created_at": datetime.now().isoformat()
            }
            
            return {
                "status": "confirmation_required",
                "confirmation_id": confirmation_id,
                "message": f"This operation requires confirmation: {tool_name}",
                "details": arguments
            }
        
        return await self.executor.execute(tool_name, arguments)
    
    async def confirm_and_execute(
        self,
        confirmation_id: str
    ) -> ExecutionResult:
        """Execute a previously confirmed operation."""
        
        if confirmation_id not in self.pending_confirmations:
            return ExecutionResult(
                status=ExecutionStatus.ERROR,
                error="Confirmation not found or expired"
            )
        
        pending = self.pending_confirmations.pop(confirmation_id)
        
        return await self.executor.execute(
            pending["tool_name"],
            pending["arguments"]
        )
```

## Parallel Tool Execution

```python
class ParallelToolExecutor:
    """Execute multiple tools in parallel."""
    
    def __init__(self, executor: ToolExecutor, max_parallel: int = 5):
        self.executor = executor
        self.semaphore = asyncio.Semaphore(max_parallel)
    
    async def execute_many(
        self,
        tool_calls: list[Dict[str, Any]]
    ) -> Dict[str, ExecutionResult]:
        """Execute multiple tool calls in parallel."""
        
        async def execute_one(call: Dict) -> tuple[str, ExecutionResult]:
            async with self.semaphore:
                result = await self.executor.execute(
                    call["name"],
                    call["arguments"]
                )
                return call["id"], result
        
        tasks = [execute_one(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            call_id: (
                result if isinstance(result, ExecutionResult)
                else ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=str(result)
                )
            )
            for call_id, result in results
        }


# Usage with OpenAI parallel function calls
async def process_tool_calls(tool_calls, executor: ParallelToolExecutor):
    """Process multiple tool calls from OpenAI."""
    
    calls = [
        {
            "id": tc.id,
            "name": tc.function.name,
            "arguments": json.loads(tc.function.arguments)
        }
        for tc in tool_calls
    ]
    
    results = await executor.execute_many(calls)
    
    # Format for OpenAI response
    return [
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result.to_llm_response()
        }
        for call_id, result in results.items()
    ]
```

## Sandboxed Execution for Code

```python
import subprocess
import tempfile
import os

class SandboxedCodeExecutor:
    """Execute code in a sandboxed environment."""
    
    def __init__(self, timeout: int = 30, max_memory_mb: int = 512):
        self.timeout = timeout
        self.max_memory = max_memory_mb
    
    async def execute_python(self, code: str) -> ExecutionResult:
        """Execute Python code in a sandbox."""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Run in subprocess with restrictions
            process = await asyncio.create_subprocess_exec(
                'python', temp_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Limit memory (Linux only)
                # preexec_fn=self._set_limits
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    error=f"Code execution timed out after {self.timeout}s"
                )
            
            if process.returncode != 0:
                return ExecutionResult(
                    status=ExecutionStatus.ERROR,
                    error=stderr.decode()[:1000]  # Limit error length
                )
            
            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                output=stdout.decode()[:10000]  # Limit output length
            )
            
        finally:
            os.unlink(temp_file)
    
    def _set_limits(self):
        """Set resource limits for subprocess (Linux)."""
        import resource
        # Set memory limit
        resource.setrlimit(
            resource.RLIMIT_AS,
            (self.max_memory * 1024 * 1024, self.max_memory * 1024 * 1024)
        )
```

## Monitoring and Observability

```python
from dataclasses import dataclass
from typing import List
import statistics

@dataclass
class ToolMetrics:
    """Metrics for a single tool."""
    name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    total_time: float = 0.0
    execution_times: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls
    
    @property
    def avg_execution_time(self) -> float:
        if not self.execution_times:
            return 0.0
        return statistics.mean(self.execution_times)
    
    @property
    def p95_execution_time(self) -> float:
        if len(self.execution_times) < 20:
            return max(self.execution_times) if self.execution_times else 0.0
        sorted_times = sorted(self.execution_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]


class ToolMonitor:
    """Monitor tool execution metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, ToolMetrics] = {}
    
    def record_execution(
        self,
        tool_name: str,
        result: ExecutionResult
    ):
        """Record a tool execution."""
        
        if tool_name not in self.metrics:
            self.metrics[tool_name] = ToolMetrics(name=tool_name)
        
        m = self.metrics[tool_name]
        m.total_calls += 1
        m.execution_times.append(result.execution_time)
        m.total_time += result.execution_time
        
        if result.status == ExecutionStatus.SUCCESS:
            m.successful_calls += 1
        else:
            m.failed_calls += 1
    
    def get_report(self) -> Dict[str, Any]:
        """Generate a metrics report."""
        return {
            name: {
                "total_calls": m.total_calls,
                "success_rate": f"{m.success_rate:.1%}",
                "avg_time": f"{m.avg_execution_time:.2f}s",
                "p95_time": f"{m.p95_execution_time:.2f}s"
            }
            for name, m in self.metrics.items()
        }
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Tool Execution & Error Handling - Summary                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Execution Pipeline:                                                     │
│    Parse → Validate → Execute → Handle Errors → Format → Return         │
│                                                                          │
│  Key Components:                                                         │
│    • Argument validation against schema                                 │
│    • Timeout handling for long operations                               │
│    • Retry logic with exponential backoff                               │
│    • Rate limiting to prevent abuse                                     │
│                                                                          │
│  Error Strategies:                                                       │
│    • Graceful degradation with fallbacks                                │
│    • Informative error messages for LLM recovery                        │
│    • Human confirmation for risky operations                            │
│                                                                          │
│  Advanced Patterns:                                                      │
│    • Parallel execution for multiple tools                              │
│    • Sandboxed code execution                                           │
│    • Monitoring and metrics collection                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Tools Lab](/learn/agents/tool-use/tools-lab) →
