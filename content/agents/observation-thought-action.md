# Observation, Thought, Action (OTA) Pattern

Master the ReAct pattern that enables structured reasoning and action in AI agents.

## Understanding the OTA Pattern

The Observation-Thought-Action pattern (also known as ReAct) provides a structured format for agent reasoning that is both effective and interpretable.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    The OTA Cycle                                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                      ┌─────────────────┐                                │
│                      │   OBSERVATION   │                                │
│                      │                 │                                │
│                      │  What I see:    │                                │
│                      │  - Tool results │                                │
│                      │  - New info     │                                │
│                      │  - Errors       │                                │
│                      └────────┬────────┘                                │
│                               │                                          │
│                               ▼                                          │
│                      ┌─────────────────┐                                │
│                      │     THOUGHT     │                                │
│                      │                 │                                │
│                      │  What I think:  │                                │
│                      │  - Analysis     │                                │
│                      │  - Planning     │                                │
│                      │  - Reasoning    │                                │
│                      └────────┬────────┘                                │
│                               │                                          │
│                               ▼                                          │
│                      ┌─────────────────┐                                │
│                      │     ACTION      │                                │
│                      │                 │                                │
│                      │  What I do:     │                                │
│                      │  - Use tool     │                                │
│                      │  - Final answer │                                │
│                      │  - Ask user     │                                │
│                      └────────┬────────┘                                │
│                               │                                          │
│                               └──────────► (Next Observation)           │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## The Three Components

### 1. Observation

The observation is what the agent perceives from its environment:

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class ObservationType(Enum):
    TOOL_RESULT = "tool_result"
    USER_INPUT = "user_input"
    SYSTEM_MESSAGE = "system_message"
    ERROR = "error"

@dataclass
class Observation:
    """An observation from the environment."""
    
    type: ObservationType
    content: str
    source: Optional[str] = None  # e.g., tool name
    metadata: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        prefix = f"[{self.source}] " if self.source else ""
        return f"Observation ({self.type.value}): {prefix}{self.content}"
    
    def to_prompt_format(self) -> str:
        """Format observation for inclusion in prompt."""
        if self.type == ObservationType.TOOL_RESULT:
            return f"OBSERVATION from {self.source}:\n{self.content}"
        elif self.type == ObservationType.ERROR:
            return f"ERROR: {self.content}"
        else:
            return f"OBSERVATION: {self.content}"
```

### 2. Thought

The thought is the agent's reasoning about what to do:

```python
@dataclass
class Thought:
    """A thought/reasoning step from the agent."""
    
    content: str
    reasoning_type: str = "general"  # analysis, planning, synthesis, etc.
    confidence: float = 1.0
    
    def __str__(self) -> str:
        return f"Thought ({self.reasoning_type}): {self.content}"
    
    def to_prompt_format(self) -> str:
        """Format thought for inclusion in prompt."""
        return f"THOUGHT: {self.content}"


# Types of thoughts an agent might have
THOUGHT_TYPES = {
    "analysis": "Breaking down the problem or information",
    "planning": "Deciding what to do next",
    "synthesis": "Combining information from multiple sources",
    "reflection": "Evaluating progress or past actions",
    "hypothesis": "Making educated guesses",
    "conclusion": "Drawing final conclusions"
}
```

### 3. Action

The action is what the agent decides to do:

```python
class ActionType(Enum):
    TOOL_CALL = "tool_call"
    FINAL_ANSWER = "final_answer"
    ASK_USER = "ask_user"
    DELEGATE = "delegate"  # For multi-agent systems

@dataclass
class Action:
    """An action the agent wants to take."""
    
    type: ActionType
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    answer: Optional[str] = None
    question: Optional[str] = None  # For ASK_USER
    
    def __str__(self) -> str:
        if self.type == ActionType.TOOL_CALL:
            return f"Action: Call {self.tool_name} with {self.tool_input}"
        elif self.type == ActionType.FINAL_ANSWER:
            return f"Action: Final Answer - {self.answer}"
        elif self.type == ActionType.ASK_USER:
            return f"Action: Ask User - {self.question}"
        return f"Action: {self.type.value}"
    
    def to_prompt_format(self) -> str:
        """Format action for inclusion in prompt."""
        if self.type == ActionType.TOOL_CALL:
            return f"ACTION: {self.tool_name}\nACTION_INPUT: {json.dumps(self.tool_input)}"
        elif self.type == ActionType.FINAL_ANSWER:
            return f"FINAL_ANSWER: {self.answer}"
        return ""
```

## Implementing the ReAct Agent

### Complete ReAct Implementation

```python
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI

class ReActAgent:
    """Agent implementing the ReAct (Reasoning + Acting) pattern."""
    
    def __init__(
        self,
        client: AsyncOpenAI,
        tools: Dict[str, callable],
        model: str = "gpt-4"
    ):
        self.client = client
        self.tools = tools
        self.model = model
        
        # Build tool descriptions for prompt
        self.tool_descriptions = self._format_tool_descriptions()
        
        self.system_prompt = f"""You are a ReAct agent that solves problems through a cycle of:
1. OBSERVATION: Information from the environment or previous actions
2. THOUGHT: Your reasoning about what you've observed and what to do
3. ACTION: The action you decide to take

Available Tools:
{self.tool_descriptions}

Format your responses EXACTLY like this:

THOUGHT: [Your detailed reasoning here]
ACTION: [tool_name OR "Final Answer"]
ACTION_INPUT: [JSON input for the tool OR your final answer text]

Rules:
- Always start with a THOUGHT explaining your reasoning
- Be specific and detailed in your thoughts
- Only use available tools
- When you have enough information, use "Final Answer" as your action
- If an action fails, analyze why in your next thought"""
    
    def _format_tool_descriptions(self) -> str:
        """Format tool descriptions for the system prompt."""
        descriptions = []
        for name, tool in self.tools.items():
            doc = tool.__doc__ or "No description"
            descriptions.append(f"- {name}: {doc}")
        return "\n".join(descriptions)
    
    async def run(self, task: str, max_steps: int = 10) -> str:
        """Run the ReAct loop on a task."""
        
        # Initialize trajectory
        trajectory: List[Dict[str, Any]] = []
        
        # Initial observation is the task itself
        current_observation = Observation(
            type=ObservationType.USER_INPUT,
            content=task
        )
        
        for step in range(max_steps):
            print(f"\n{'='*50}")
            print(f"Step {step + 1}")
            print('='*50)
            
            # Get thought and action from LLM
            thought, action = await self._reason(current_observation, trajectory)
            
            print(f"\n💭 THOUGHT: {thought.content}")
            print(f"\n🎯 ACTION: {action}")
            
            # Record in trajectory
            trajectory.append({
                "observation": current_observation,
                "thought": thought,
                "action": action
            })
            
            # Check if we have a final answer
            if action.type == ActionType.FINAL_ANSWER:
                print(f"\n✅ FINAL ANSWER: {action.answer}")
                return action.answer
            
            # Execute the action
            if action.type == ActionType.TOOL_CALL:
                result = await self._execute_tool(action.tool_name, action.tool_input)
                current_observation = Observation(
                    type=ObservationType.TOOL_RESULT if result.success else ObservationType.ERROR,
                    content=result.content,
                    source=action.tool_name
                )
                print(f"\n👁 OBSERVATION: {current_observation.content[:200]}...")
            
            elif action.type == ActionType.ASK_USER:
                # In a real system, you'd wait for user input
                current_observation = Observation(
                    type=ObservationType.USER_INPUT,
                    content="[User would respond here]"
                )
        
        return "Maximum steps reached without finding an answer."
    
    async def _reason(
        self,
        observation: Observation,
        trajectory: List[Dict[str, Any]]
    ) -> Tuple[Thought, Action]:
        """Generate thought and action based on current state."""
        
        # Build messages from trajectory
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for entry in trajectory:
            # Add the observation-thought-action sequence
            obs = entry["observation"]
            thought = entry["thought"]
            action = entry["action"]
            
            messages.append({
                "role": "user",
                "content": obs.to_prompt_format()
            })
            messages.append({
                "role": "assistant",
                "content": f"{thought.to_prompt_format()}\n{action.to_prompt_format()}"
            })
        
        # Add current observation
        messages.append({
            "role": "user",
            "content": observation.to_prompt_format()
        })
        
        # Get LLM response
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content
        
        # Parse response
        thought, action = self._parse_react_response(response_text)
        
        return thought, action
    
    def _parse_react_response(self, response: str) -> Tuple[Thought, Action]:
        """Parse the LLM response into Thought and Action."""
        
        # Extract thought
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response, re.DOTALL)
        thought_content = thought_match.group(1).strip() if thought_match else ""
        thought = Thought(content=thought_content)
        
        # Extract action
        action_match = re.search(r'ACTION:\s*(.+?)(?=ACTION_INPUT:|$)', response, re.DOTALL)
        action_name = action_match.group(1).strip() if action_match else ""
        
        # Extract action input
        input_match = re.search(r'ACTION_INPUT:\s*(.+?)$', response, re.DOTALL)
        action_input_str = input_match.group(1).strip() if input_match else ""
        
        # Determine action type
        if action_name.lower() == "final answer":
            action = Action(
                type=ActionType.FINAL_ANSWER,
                answer=action_input_str
            )
        else:
            # Parse JSON input
            try:
                tool_input = json.loads(action_input_str)
            except json.JSONDecodeError:
                tool_input = {"input": action_input_str}
            
            action = Action(
                type=ActionType.TOOL_CALL,
                tool_name=action_name,
                tool_input=tool_input
            )
        
        return thought, action
    
    async def _execute_tool(
        self, 
        tool_name: str, 
        tool_input: Dict[str, Any]
    ) -> 'ToolResult':
        """Execute a tool and return the result."""
        
        if tool_name not in self.tools:
            return ToolResult(
                success=False,
                content=f"Unknown tool: {tool_name}. Available: {list(self.tools.keys())}"
            )
        
        try:
            tool = self.tools[tool_name]
            result = await tool(**tool_input) if asyncio.iscoroutinefunction(tool) else tool(**tool_input)
            return ToolResult(success=True, content=str(result))
        except Exception as e:
            return ToolResult(success=False, content=f"Error: {str(e)}")


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    content: str
```

## ReAct Trace Example

Here's what a complete ReAct trace looks like:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ReAct Trace Example                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Task: "Who is the CEO of the company that makes the iPhone,            │
│         and what is their net worth?"                                   │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  OBSERVATION [Step 1]: User query about iPhone maker's CEO net worth    │
│                                                                          │
│  THOUGHT [Step 1]: I need to find out who makes the iPhone first.       │
│  Apple makes the iPhone. But I should verify who the current CEO is     │
│  and then find their net worth. Let me search for Apple's CEO.          │
│                                                                          │
│  ACTION [Step 1]: search                                                │
│  ACTION_INPUT: {"query": "Apple Inc CEO 2024"}                          │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  OBSERVATION [Step 2]: Tim Cook is the CEO of Apple Inc since 2011     │
│                                                                          │
│  THOUGHT [Step 2]: Tim Cook is Apple's CEO. Now I need to find his     │
│  current net worth. Let me search for that information.                 │
│                                                                          │
│  ACTION [Step 2]: search                                                │
│  ACTION_INPUT: {"query": "Tim Cook net worth 2024"}                     │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  OBSERVATION [Step 3]: Tim Cook net worth estimated at $1.8 billion    │
│                                                                          │
│  THOUGHT [Step 3]: I now have all the information needed. Apple makes  │
│  the iPhone, Tim Cook is the CEO, and his net worth is ~$1.8 billion.  │
│  I can provide the final answer.                                        │
│                                                                          │
│  ACTION [Step 3]: Final Answer                                          │
│  ACTION_INPUT: The company that makes the iPhone is Apple Inc.          │
│  The CEO of Apple is Tim Cook, who has been in the role since 2011.    │
│  His estimated net worth is approximately $1.8 billion.                 │
│                                                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                          │
│  ✓ COMPLETED in 3 steps                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Advanced OTA Patterns

### Self-Reflection in Thoughts

```python
class ReflectiveReActAgent(ReActAgent):
    """ReAct agent with self-reflection capabilities."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.system_prompt += """

When thinking, consider:
1. What have I learned so far?
2. Am I making progress toward the goal?
3. Should I try a different approach?
4. What could go wrong with my current plan?
5. Do I have enough information to answer?"""
    
    async def _reason_with_reflection(
        self,
        observation: Observation,
        trajectory: List[Dict[str, Any]]
    ) -> Tuple[Thought, Action]:
        """Enhanced reasoning with explicit reflection."""
        
        # If we have multiple steps, add reflection
        if len(trajectory) >= 2:
            reflection_prompt = self._generate_reflection_prompt(trajectory)
            observation = Observation(
                type=ObservationType.SYSTEM_MESSAGE,
                content=f"{observation.content}\n\nREFLECTION PROMPT: {reflection_prompt}"
            )
        
        return await self._reason(observation, trajectory)
    
    def _generate_reflection_prompt(
        self, 
        trajectory: List[Dict[str, Any]]
    ) -> str:
        """Generate a reflection prompt based on trajectory."""
        
        questions = [
            "Are you making progress?",
            "Have any of your actions failed or given unexpected results?",
            "Do you need to try a different approach?",
            "Do you have enough information to provide a final answer?"
        ]
        
        return " ".join(questions)
```

### Chain-of-Thought in Actions

```python
def enhance_thought_prompt(base_prompt: str) -> str:
    """Add chain-of-thought enhancement to the prompt."""
    
    cot_addition = """

When writing your THOUGHT, use structured reasoning:

1. **Summarize**: What do I know from observations?
2. **Analyze**: What does this information mean?
3. **Plan**: What should I do next and why?
4. **Consider**: What could go wrong?
5. **Decide**: What's my best action?

Example:
THOUGHT: 
- Summary: The search returned information about Python packages for web scraping.
- Analysis: BeautifulSoup and Scrapy are the most recommended options.
- Plan: I should compare their features to recommend the best one.
- Consideration: The user didn't specify their use case, which affects the recommendation.
- Decision: I'll ask about their specific needs before recommending."""
    
    return base_prompt + cot_addition
```

## Best Practices for OTA

```yaml
observation_best_practices:
  - "Include all relevant information from tool results"
  - "Note errors or failures explicitly"
  - "Preserve important details for future reference"
  - "Format structured data clearly"

thought_best_practices:
  - "Always explain reasoning before acting"
  - "Reference specific observations"
  - "Consider multiple approaches"
  - "Acknowledge uncertainty when present"
  - "Plan multiple steps ahead when possible"

action_best_practices:
  - "Choose the most appropriate tool for the task"
  - "Provide complete and valid tool inputs"
  - "Know when to stop and give a final answer"
  - "Handle errors by adjusting approach"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│             Observation-Thought-Action - Summary                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  The Pattern:                                                            │
│    OBSERVE → THINK → ACT → (repeat)                                     │
│                                                                          │
│  Components:                                                             │
│    • Observation: What the agent perceives                              │
│    • Thought: Reasoning and planning                                    │
│    • Action: Tool use or final answer                                   │
│                                                                          │
│  Key Benefits:                                                           │
│    • Transparent reasoning                                              │
│    • Debuggable traces                                                  │
│    • Flexible problem-solving                                           │
│    • Natural error recovery                                             │
│                                                                          │
│  Implementation Tips:                                                    │
│    • Use structured parsing for reliability                             │
│    • Add reflection for complex tasks                                   │
│    • Include chain-of-thought in prompts                                │
│    • Handle parsing failures gracefully                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Memory and State Management](/learn/agents/agent-loop/memory-state) →
