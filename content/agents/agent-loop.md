# The Agent Loop

Understand the core execution loop that drives AI agents through reasoning and action cycles.

## What is the Agent Loop?

The agent loop is the fundamental control flow that enables an agent to iteratively reason, act, observe, and adapt until a task is complete.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       The Agent Loop                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌─────────────┐                                 │
│                         │   START     │                                 │
│                         │   (Task)    │                                 │
│                         └──────┬──────┘                                 │
│                                │                                         │
│                                ▼                                         │
│                    ┌───────────────────────┐                            │
│                    │       REASON          │                            │
│                    │  (Analyze situation)  │◄──────────┐                │
│                    └───────────┬───────────┘           │                │
│                                │                       │                │
│                                ▼                       │                │
│                    ┌───────────────────────┐           │                │
│                    │       DECIDE          │           │                │
│                    │  (Choose action)      │           │                │
│                    └───────────┬───────────┘           │                │
│                                │                       │                │
│                        ┌───────┴───────┐               │                │
│                        ▼               ▼               │                │
│               ┌────────────┐   ┌────────────┐         │                │
│               │   ACTION   │   │    DONE    │         │                │
│               │ (Use tool) │   │  (Answer)  │         │                │
│               └─────┬──────┘   └────────────┘         │                │
│                     │                                  │                │
│                     ▼                                  │                │
│               ┌────────────┐                          │                │
│               │  OBSERVE   │                          │                │
│               │  (Result)  │──────────────────────────┘                │
│               └────────────┘                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Components of the Loop

### 1. Initialization

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum

class AgentState(Enum):
    """Possible states of the agent."""
    IDLE = "idle"
    REASONING = "reasoning"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class AgentContext:
    """Context maintained throughout the agent loop."""
    
    # Original task
    task: str
    
    # Current state
    state: AgentState = AgentState.IDLE
    
    # Accumulated history
    thoughts: List[str] = field(default_factory=list)
    actions: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    
    # Loop control
    iteration: int = 0
    max_iterations: int = 10
    
    # Result
    final_answer: Optional[str] = None
    error: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if the agent has finished."""
        return self.state in [AgentState.COMPLETED, AgentState.ERROR]
    
    @property
    def should_continue(self) -> bool:
        """Check if the loop should continue."""
        return not self.is_complete and self.iteration < self.max_iterations
```

### 2. The Main Loop Implementation

```python
from openai import AsyncOpenAI
import json

class AgentLoop:
    """The core agent execution loop."""
    
    def __init__(
        self,
        llm_client: AsyncOpenAI,
        tools: Dict[str, callable],
        model: str = "gpt-4"
    ):
        self.client = llm_client
        self.tools = tools
        self.model = model
        
        self.system_prompt = """You are an AI agent that solves tasks step by step.

For each step, you must:
1. THINK: Analyze the current situation and what you know
2. DECIDE: Choose to either use a tool OR provide a final answer

When using a tool, format your response as:
THOUGHT: [Your reasoning]
ACTION: [tool_name]
ACTION_INPUT: [JSON input for the tool]

When you have the final answer:
THOUGHT: [Your final reasoning]
FINAL_ANSWER: [Your complete answer to the user's task]

Always think before acting. Use observations from previous actions to inform your decisions."""

    async def run(self, task: str) -> str:
        """Run the agent loop until completion."""
        
        context = AgentContext(task=task)
        
        while context.should_continue:
            context.iteration += 1
            
            try:
                # Step 1: Reason and decide
                context.state = AgentState.REASONING
                response = await self._get_llm_response(context)
                
                # Step 2: Parse the response
                parsed = self._parse_response(response)
                
                # Record the thought
                if parsed.get("thought"):
                    context.thoughts.append(parsed["thought"])
                
                # Step 3: Check if we have a final answer
                if parsed.get("final_answer"):
                    context.final_answer = parsed["final_answer"]
                    context.state = AgentState.COMPLETED
                    break
                
                # Step 4: Execute action if specified
                if parsed.get("action"):
                    context.state = AgentState.ACTING
                    action_result = await self._execute_action(
                        parsed["action"],
                        parsed.get("action_input", {})
                    )
                    
                    context.actions.append({
                        "action": parsed["action"],
                        "input": parsed["action_input"],
                        "result": action_result
                    })
                    
                    # Step 5: Observe result
                    context.state = AgentState.OBSERVING
                    context.observations.append(action_result)
                
            except Exception as e:
                context.state = AgentState.ERROR
                context.error = str(e)
                break
        
        # Handle max iterations reached
        if not context.is_complete:
            context.final_answer = self._synthesize_partial_answer(context)
            context.state = AgentState.COMPLETED
        
        return context.final_answer
    
    async def _get_llm_response(self, context: AgentContext) -> str:
        """Get LLM response given current context."""
        
        # Build the prompt with full history
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Task: {context.task}"}
        ]
        
        # Add history
        for i, (thought, action, obs) in enumerate(zip(
            context.thoughts,
            context.actions,
            context.observations
        )):
            messages.append({
                "role": "assistant",
                "content": f"THOUGHT: {thought}\nACTION: {action['action']}\nACTION_INPUT: {json.dumps(action['input'])}"
            })
            messages.append({
                "role": "user",
                "content": f"OBSERVATION: {obs}"
            })
        
        # Request next step
        if context.iteration > 1:
            messages.append({
                "role": "user",
                "content": "Continue with your next thought and action, or provide the final answer if you have enough information."
            })
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM response into components."""
        
        result = {}
        
        # Extract thought
        if "THOUGHT:" in response:
            thought_start = response.index("THOUGHT:") + len("THOUGHT:")
            thought_end = len(response)
            
            for marker in ["ACTION:", "FINAL_ANSWER:"]:
                if marker in response:
                    marker_pos = response.index(marker)
                    if marker_pos > thought_start:
                        thought_end = min(thought_end, marker_pos)
            
            result["thought"] = response[thought_start:thought_end].strip()
        
        # Extract final answer
        if "FINAL_ANSWER:" in response:
            answer_start = response.index("FINAL_ANSWER:") + len("FINAL_ANSWER:")
            result["final_answer"] = response[answer_start:].strip()
        
        # Extract action
        elif "ACTION:" in response:
            action_start = response.index("ACTION:") + len("ACTION:")
            action_end = response.index("ACTION_INPUT:") if "ACTION_INPUT:" in response else len(response)
            result["action"] = response[action_start:action_end].strip()
            
            # Extract action input
            if "ACTION_INPUT:" in response:
                input_start = response.index("ACTION_INPUT:") + len("ACTION_INPUT:")
                input_str = response[input_start:].strip()
                
                try:
                    result["action_input"] = json.loads(input_str)
                except json.JSONDecodeError:
                    result["action_input"] = {"raw_input": input_str}
        
        return result
    
    async def _execute_action(
        self, 
        action: str, 
        action_input: Dict[str, Any]
    ) -> str:
        """Execute an action using the appropriate tool."""
        
        if action not in self.tools:
            return f"Error: Unknown action '{action}'. Available actions: {list(self.tools.keys())}"
        
        try:
            tool = self.tools[action]
            result = await tool(**action_input) if asyncio.iscoroutinefunction(tool) else tool(**action_input)
            return str(result)
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    def _synthesize_partial_answer(self, context: AgentContext) -> str:
        """Synthesize an answer from partial results."""
        
        summary = f"I attempted to complete the task but reached the iteration limit.\n\n"
        summary += f"Task: {context.task}\n\n"
        summary += f"Progress made:\n"
        
        for i, (action, obs) in enumerate(zip(context.actions, context.observations)):
            summary += f"{i+1}. {action['action']}: {obs[:200]}...\n"
        
        return summary
```

## Loop Flow Visualization

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Agent Loop Timeline                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Iteration 1                                                             │
│  ──────────                                                              │
│  User: "What's the weather in Paris and should I bring an umbrella?"    │
│      │                                                                   │
│      ▼                                                                   │
│  THOUGHT: I need to check the weather in Paris first                    │
│  ACTION: weather_search                                                  │
│  INPUT: {"city": "Paris"}                                               │
│      │                                                                   │
│      ▼                                                                   │
│  OBSERVATION: "Paris: 18°C, 60% chance of rain"                         │
│                                                                          │
│  Iteration 2                                                             │
│  ──────────                                                              │
│  THOUGHT: There's a 60% chance of rain, which is significant            │
│  FINAL_ANSWER: "The weather in Paris is 18°C with a 60% chance of       │
│                rain. Yes, I recommend bringing an umbrella."            │
│                                                                          │
│  ✓ COMPLETED (2 iterations)                                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Handling Loop Termination

### Termination Conditions

```python
class LoopTermination:
    """Handles various loop termination scenarios."""
    
    @staticmethod
    def check_termination(context: AgentContext) -> tuple[bool, str]:
        """Check if the loop should terminate and why."""
        
        # 1. Task completed successfully
        if context.final_answer:
            return True, "completed"
        
        # 2. Maximum iterations reached
        if context.iteration >= context.max_iterations:
            return True, "max_iterations"
        
        # 3. Error occurred
        if context.error:
            return True, "error"
        
        # 4. Stuck in loop (same action repeated)
        if len(context.actions) >= 3:
            last_three = context.actions[-3:]
            if all(a["action"] == last_three[0]["action"] for a in last_three):
                if all(a["input"] == last_three[0]["input"] for a in last_three):
                    return True, "stuck_loop"
        
        # 5. No progress being made
        if len(context.observations) >= 2:
            if context.observations[-1] == context.observations[-2]:
                return True, "no_progress"
        
        return False, "continue"
```

### Graceful Degradation

```python
class ResilientAgentLoop(AgentLoop):
    """Agent loop with enhanced error recovery."""
    
    async def run(self, task: str) -> str:
        """Run with error recovery."""
        
        context = AgentContext(task=task)
        consecutive_errors = 0
        max_consecutive_errors = 3
        
        while context.should_continue:
            context.iteration += 1
            
            try:
                response = await self._get_llm_response(context)
                parsed = self._parse_response(response)
                
                # Reset error counter on success
                consecutive_errors = 0
                
                if parsed.get("thought"):
                    context.thoughts.append(parsed["thought"])
                
                if parsed.get("final_answer"):
                    context.final_answer = parsed["final_answer"]
                    context.state = AgentState.COMPLETED
                    break
                
                if parsed.get("action"):
                    action_result = await self._execute_action_safe(
                        parsed["action"],
                        parsed.get("action_input", {}),
                        context
                    )
                    
                    context.actions.append({
                        "action": parsed["action"],
                        "input": parsed["action_input"],
                        "result": action_result
                    })
                    context.observations.append(action_result)
                    
            except Exception as e:
                consecutive_errors += 1
                error_msg = f"Error in iteration {context.iteration}: {str(e)}"
                
                if consecutive_errors >= max_consecutive_errors:
                    context.error = f"Too many consecutive errors: {error_msg}"
                    context.state = AgentState.ERROR
                    break
                
                # Record error as observation to help LLM recover
                context.observations.append(f"SYSTEM ERROR: {error_msg}. Please try a different approach.")
        
        return self._format_final_response(context)
    
    async def _execute_action_safe(
        self,
        action: str,
        action_input: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """Execute action with safety checks."""
        
        # Check for potentially dangerous patterns
        if action == "execute_code":
            if self._contains_dangerous_code(action_input.get("code", "")):
                return "Error: Code contains potentially dangerous operations and was not executed."
        
        # Add timeout protection
        try:
            result = await asyncio.wait_for(
                self._execute_action(action, action_input),
                timeout=30.0
            )
            return result
        except asyncio.TimeoutError:
            return f"Error: Action '{action}' timed out after 30 seconds."
    
    def _contains_dangerous_code(self, code: str) -> bool:
        """Check for dangerous code patterns."""
        dangerous_patterns = [
            "os.system", "subprocess", "eval(", "exec(",
            "rm -rf", "del /", "format c:",
            "__import__", "open(", "write("
        ]
        return any(pattern in code.lower() for pattern in dangerous_patterns)
    
    def _format_final_response(self, context: AgentContext) -> str:
        """Format the final response based on context state."""
        
        if context.state == AgentState.COMPLETED and context.final_answer:
            return context.final_answer
        
        if context.state == AgentState.ERROR:
            return f"I encountered an error while processing your request: {context.error}\n\nHere's what I was able to determine:\n{self._summarize_progress(context)}"
        
        return f"I've gathered some information but couldn't reach a definitive answer:\n{self._summarize_progress(context)}"
    
    def _summarize_progress(self, context: AgentContext) -> str:
        """Summarize the progress made."""
        
        if not context.observations:
            return "No progress was made."
        
        summary = []
        for i, obs in enumerate(context.observations):
            summary.append(f"Step {i+1}: {obs[:300]}")
        
        return "\n".join(summary)
```

## Streaming the Agent Loop

For better UX, stream the agent's progress:

```python
from typing import AsyncGenerator

class StreamingAgentLoop(AgentLoop):
    """Agent loop that streams progress updates."""
    
    async def run_stream(
        self, 
        task: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Run the loop and yield progress updates."""
        
        context = AgentContext(task=task)
        
        yield {
            "type": "start",
            "task": task,
            "timestamp": datetime.now().isoformat()
        }
        
        while context.should_continue:
            context.iteration += 1
            
            yield {
                "type": "iteration_start",
                "iteration": context.iteration
            }
            
            try:
                response = await self._get_llm_response(context)
                parsed = self._parse_response(response)
                
                if parsed.get("thought"):
                    context.thoughts.append(parsed["thought"])
                    yield {
                        "type": "thought",
                        "content": parsed["thought"]
                    }
                
                if parsed.get("final_answer"):
                    context.final_answer = parsed["final_answer"]
                    context.state = AgentState.COMPLETED
                    yield {
                        "type": "answer",
                        "content": parsed["final_answer"]
                    }
                    break
                
                if parsed.get("action"):
                    yield {
                        "type": "action",
                        "action": parsed["action"],
                        "input": parsed["action_input"]
                    }
                    
                    result = await self._execute_action(
                        parsed["action"],
                        parsed.get("action_input", {})
                    )
                    
                    context.actions.append({
                        "action": parsed["action"],
                        "input": parsed["action_input"],
                        "result": result
                    })
                    context.observations.append(result)
                    
                    yield {
                        "type": "observation",
                        "content": result
                    }
                    
            except Exception as e:
                yield {
                    "type": "error",
                    "message": str(e)
                }
                context.state = AgentState.ERROR
                break
        
        yield {
            "type": "complete",
            "iterations": context.iteration,
            "success": context.state == AgentState.COMPLETED
        }
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent Loop - Summary                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Core Cycle:  REASON → DECIDE → ACT → OBSERVE → (repeat)               │
│                                                                          │
│  Key Components:                                                         │
│    • AgentContext: Maintains state across iterations                    │
│    • LLM Response Parsing: Extracts thoughts, actions, answers          │
│    • Tool Execution: Runs actions and captures results                  │
│    • Termination Logic: Knows when to stop                              │
│                                                                          │
│  Termination Conditions:                                                 │
│    • Final answer provided                                              │
│    • Maximum iterations reached                                         │
│    • Error occurred                                                     │
│    • Stuck in a loop                                                    │
│                                                                          │
│  Best Practices:                                                         │
│    • Stream progress for better UX                                      │
│    • Handle errors gracefully                                           │
│    • Detect and break loops                                             │
│    • Maintain full history for context                                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Observation, Thought, Action Pattern](/learn/agents/agent-loop/ota-pattern) →
