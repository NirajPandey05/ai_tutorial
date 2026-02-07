# Agent Architectures

Explore the major architectural patterns for building AI agents, from simple ReAct to sophisticated planning systems.

## Overview of Agent Architectures

Different tasks require different agent architectures. Understanding these patterns helps you choose the right approach for your use case.

### Architecture Selection Guide

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Agent Architecture Selection Matrix                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│ ARCHITECTURE    │ BEST FOR              │ COMPLEXITY │ SUCCESS RATE     │
│ ────────────────┼───────────────────────┼────────────┼──────────────── │
│ Simple Tool Use │ • Basic tasks         │ Low        │ 70-80%          │
│                 │ • Single action       │            │                  │
│                 │ • Quick responses     │            │                  │
│ ────────────────┼───────────────────────┼────────────┼──────────────── │
│ ReAct           │ • Multi-step tasks    │ Medium     │ 80-85%          │
│                 │ • Reasoning needed    │            │                  │
│                 │ • Research queries    │            │                  │
│ ────────────────┼───────────────────────┼────────────┼──────────────── │
│ Plan-Execute    │ • Complex goals       │ Medium-High│ 85-90%          │
│                 │ • Multi-phase tasks   │            │                  │
│                 │ • Resource dependent  │            │                  │
│ ────────────────┼───────────────────────┼────────────┼──────────────── │
│ Multi-Agent     │ • Parallel tasks      │ High       │ 75-88%          │
│                 │ • Specialization      │            │                  │
│                 │ • Complex workflows   │            │                  │
│ ────────────────┼───────────────────────┼────────────┼──────────────── │
│ LATS/Tree Search│ • Optimization needed │ Very High  │ 90%+            │
│                 │ • Critical decisions  │            │                  │
│                 │ • Verification needed │            │                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Core Architecture Components

All agent architectures share common elements:

```yaml
agent_components:
  1_perception: "Observe environment and parse feedback"
  2_reasoning: "Think about what to do next"
  3_planning: "Decide which action to take"
  4_action: "Execute tools or interactions"
  5_evaluation: "Check if goal is achieved"
  6_memory: "Learn from past interactions"
```

---

## 1. ReAct (Reasoning + Acting)

The most fundamental and widely-used agent architecture.

### How ReAct Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ReAct Pattern                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐                                                       │
│   │   INPUT     │  "What's the population of the capital of France?"   │
│   └──────┬──────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                       │
│   │  THOUGHT    │  "I need to find the capital of France first"        │
│   └──────┬──────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                       │
│   │   ACTION    │  search("capital of France")                         │
│   └──────┬──────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                       │
│   │ OBSERVATION │  "Paris is the capital of France"                    │
│   └──────┬──────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                       │
│   │  THOUGHT    │  "Now I need to find Paris's population"             │
│   └──────┬──────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                       │
│   │   ACTION    │  search("population of Paris")                       │
│   └──────┬──────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                       │
│   │ OBSERVATION │  "Paris has a population of 2.1 million"             │
│   └──────┬──────┘                                                       │
│          │                                                               │
│          ▼                                                               │
│   ┌─────────────┐                                                       │
│   │   ANSWER    │  "The population of Paris is 2.1 million"            │
│   └─────────────┘                                                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### ReAct Implementation

```python
from typing import List, Dict, Any

class ReActAgent:
    """Simple ReAct agent implementation."""
    
    def __init__(self, llm, tools: Dict[str, callable]):
        self.llm = llm
        self.tools = tools
        self.max_iterations = 10
    
    def run(self, task: str) -> str:
        """Execute the ReAct loop."""
        
        # Build initial prompt
        prompt = self._build_prompt(task)
        trajectory = []
        
        for i in range(self.max_iterations):
            # Get LLM response (thought + action)
            response = self.llm.generate(prompt)
            
            # Parse thought and action
            thought, action, action_input = self._parse_response(response)
            
            trajectory.append({
                "thought": thought,
                "action": action,
                "action_input": action_input
            })
            
            # Check if done
            if action == "Final Answer":
                return action_input
            
            # Execute action and get observation
            observation = self._execute_action(action, action_input)
            trajectory[-1]["observation"] = observation
            
            # Update prompt with new observation
            prompt = self._update_prompt(prompt, thought, action, 
                                         action_input, observation)
        
        return "Max iterations reached without answer"
    
    def _execute_action(self, action: str, action_input: str) -> str:
        """Execute a tool and return observation."""
        if action not in self.tools:
            return f"Unknown tool: {action}"
        
        try:
            return self.tools[action](action_input)
        except Exception as e:
            return f"Error executing {action}: {str(e)}"
    
    def _build_prompt(self, task: str) -> str:
        """Build the initial ReAct prompt."""
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__}" 
            for name, func in self.tools.items()
        ])
        
        return f"""Answer the following question using the available tools.

Available Tools:
{tool_descriptions}

Use the following format:

Thought: [Your reasoning about what to do next]
Action: [The tool to use]
Action Input: [The input to the tool]
Observation: [The result from the tool]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have enough information to answer
Action: Final Answer
Action Input: [Your final answer]

Question: {task}

Thought:"""
```

### ReAct Pros & Cons

```yaml
pros:
  - "Simple and intuitive"
  - "Interpretable reasoning trace"
  - "Works well for most tasks"
  - "Easy to debug"

cons:
  - "Can get stuck in loops"
  - "No explicit planning"
  - "Greedy decision making"
  - "May be inefficient for complex tasks"
```

## 2. Plan-and-Execute

Separates planning from execution for more complex tasks.

### How Plan-and-Execute Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Plan-and-Execute Pattern                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                        PLANNER                               │       │
│   │                                                              │       │
│   │  Input: "Create a blog post about AI trends"                │       │
│   │                                                              │       │
│   │  Plan:                                                       │       │
│   │    1. Research current AI trends                            │       │
│   │    2. Outline the blog structure                            │       │
│   │    3. Write introduction                                    │       │
│   │    4. Write main sections                                   │       │
│   │    5. Write conclusion                                      │       │
│   │    6. Review and edit                                       │       │
│   └─────────────────────────────────────────────────────────────┘       │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                       EXECUTOR                               │       │
│   │                                                              │       │
│   │  Step 1: Research ──► [web_search, read_articles]           │       │
│   │      ↓                                                       │       │
│   │  Step 2: Outline ──► [create_outline]                       │       │
│   │      ↓                                                       │       │
│   │  Step 3: Intro ──► [write_section]                          │       │
│   │      ↓                                                       │       │
│   │  Step 4: Main ──► [write_section] × 3                       │       │
│   │      ↓                                                       │       │
│   │  Step 5: Conclusion ──► [write_section]                     │       │
│   │      ↓                                                       │       │
│   │  Step 6: Review ──► [review_document]                       │       │
│   └─────────────────────────────────────────────────────────────┘       │
│                               │                                          │
│                               ▼                                          │
│   ┌─────────────────────────────────────────────────────────────┐       │
│   │                       REPLANNER                              │       │
│   │                                                              │       │
│   │  • Check progress against plan                              │       │
│   │  • Adjust plan if needed                                    │       │
│   │  • Handle execution failures                                │       │
│   └─────────────────────────────────────────────────────────────┘       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Plan-and-Execute Implementation

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Step:
    """A single step in the plan."""
    description: str
    status: str = "pending"  # pending, in_progress, completed, failed
    result: Optional[str] = None

class PlanAndExecuteAgent:
    """Agent that plans first, then executes."""
    
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools
    
    async def run(self, task: str) -> str:
        # Phase 1: Create plan
        plan = await self._create_plan(task)
        
        # Phase 2: Execute each step
        for step in plan:
            step.status = "in_progress"
            
            try:
                result = await self._execute_step(step, plan)
                step.result = result
                step.status = "completed"
            except Exception as e:
                step.status = "failed"
                # Optionally replan
                plan = await self._replan(task, plan, step, str(e))
        
        # Phase 3: Synthesize final answer
        return await self._synthesize_answer(task, plan)
    
    async def _create_plan(self, task: str) -> List[Step]:
        """Use planner LLM to create a step-by-step plan."""
        prompt = f"""Create a detailed plan to accomplish this task:

Task: {task}

Break this down into clear, actionable steps.
Each step should be specific and achievable.

Format your response as a numbered list:
1. [First step]
2. [Second step]
...
"""
        response = await self.planner.generate(prompt)
        return self._parse_plan(response)
    
    async def _execute_step(self, step: Step, full_plan: List[Step]) -> str:
        """Execute a single step using the executor LLM."""
        # Get context from completed steps
        context = self._get_execution_context(full_plan)
        
        prompt = f"""Execute this step:

Step: {step.description}

Previous context:
{context}

Available tools: {list(self.tools.keys())}

Execute this step and provide the result.
"""
        # Use ReAct-style execution for the step
        return await self._react_execute(prompt)
    
    async def _replan(
        self, 
        task: str, 
        current_plan: List[Step],
        failed_step: Step,
        error: str
    ) -> List[Step]:
        """Replan after a failure."""
        prompt = f"""The following plan encountered an issue:

Original task: {task}

Current plan:
{self._format_plan(current_plan)}

Failed step: {failed_step.description}
Error: {error}

Please revise the plan to work around this issue.
"""
        response = await self.planner.generate(prompt)
        return self._parse_plan(response)
```

### Plan-and-Execute Pros & Cons

```yaml
pros:
  - "Better for complex, multi-step tasks"
  - "Can adapt plan based on results"
  - "More strategic thinking"
  - "Handles dependencies between steps"

cons:
  - "Initial planning takes time"
  - "Plan may become stale"
  - "Requires good task decomposition"
  - "More complex to implement"
```

## 3. Reflexion

Adds self-reflection and learning from mistakes.

### How Reflexion Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Reflexion Pattern                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌─────────────┐                                 │
│                         │    TASK     │                                 │
│                         └──────┬──────┘                                 │
│                                │                                         │
│                                ▼                                         │
│   ┌────────────────────────────────────────────────────────────┐        │
│   │                      ACTOR                                  │        │
│   │            (Attempts to solve the task)                     │        │
│   └────────────────────────────┬───────────────────────────────┘        │
│                                │                                         │
│                                ▼                                         │
│   ┌────────────────────────────────────────────────────────────┐        │
│   │                    EVALUATOR                                │        │
│   │            (Checks if solution is correct)                  │        │
│   └────────────────────────────┬───────────────────────────────┘        │
│                                │                                         │
│                     ┌──────────┴──────────┐                             │
│                     │                     │                             │
│                     ▼                     ▼                             │
│              ┌──────────┐          ┌──────────┐                         │
│              │ SUCCESS  │          │ FAILURE  │                         │
│              └──────────┘          └────┬─────┘                         │
│                                         │                                │
│                                         ▼                                │
│   ┌────────────────────────────────────────────────────────────┐        │
│   │                    SELF-REFLECTION                          │        │
│   │                                                             │        │
│   │  "I failed because I didn't consider edge cases..."        │        │
│   │  "Next time I should verify the input format..."           │        │
│   │  "The error was in step 3 where I assumed..."              │        │
│   └────────────────────────────┬───────────────────────────────┘        │
│                                │                                         │
│                                │  (Store reflection in memory)          │
│                                │                                         │
│                                ▼                                         │
│   ┌────────────────────────────────────────────────────────────┐        │
│   │                      ACTOR                                  │        │
│   │         (Retry with reflection in context)                  │        │
│   └────────────────────────────────────────────────────────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Reflexion Implementation

```python
class ReflexionAgent:
    """Agent that learns from its mistakes."""
    
    def __init__(self, actor_llm, evaluator_llm, max_retries: int = 3):
        self.actor = actor_llm
        self.evaluator = evaluator_llm
        self.max_retries = max_retries
        self.reflections: List[str] = []
    
    async def run(self, task: str) -> str:
        for attempt in range(self.max_retries):
            # Actor attempts the task
            solution = await self._act(task)
            
            # Evaluator checks the solution
            is_correct, feedback = await self._evaluate(task, solution)
            
            if is_correct:
                return solution
            
            # Self-reflect on failure
            reflection = await self._reflect(task, solution, feedback)
            self.reflections.append(reflection)
            
            print(f"Attempt {attempt + 1} failed. Reflection: {reflection}")
        
        return f"Failed after {self.max_retries} attempts"
    
    async def _act(self, task: str) -> str:
        """Generate a solution, using past reflections."""
        reflections_text = "\n".join([
            f"- {r}" for r in self.reflections
        ]) if self.reflections else "None"
        
        prompt = f"""Solve this task:

Task: {task}

Lessons from previous attempts:
{reflections_text}

Provide your solution:"""
        
        return await self.actor.generate(prompt)
    
    async def _evaluate(self, task: str, solution: str) -> tuple[bool, str]:
        """Evaluate if the solution is correct."""
        prompt = f"""Evaluate this solution:

Task: {task}

Solution: {solution}

Is this solution correct? Explain why or why not.
Format: CORRECT/INCORRECT: [explanation]"""
        
        response = await self.evaluator.generate(prompt)
        is_correct = response.startswith("CORRECT")
        return is_correct, response
    
    async def _reflect(self, task: str, solution: str, feedback: str) -> str:
        """Generate a reflection on what went wrong."""
        prompt = f"""Reflect on this failed attempt:

Task: {task}
My solution: {solution}
Feedback: {feedback}

What went wrong? What should I do differently next time?
Be specific and actionable."""
        
        return await self.actor.generate(prompt)
```

## 4. LATS (Language Agent Tree Search)

Uses tree search for more thorough exploration.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LATS (Tree Search) Pattern                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                              [Start]                                     │
│                                 │                                        │
│                    ┌────────────┼────────────┐                          │
│                    │            │            │                          │
│                    ▼            ▼            ▼                          │
│               [Action A]   [Action B]   [Action C]                      │
│               score: 0.7   score: 0.9   score: 0.5                      │
│                    │            │            │                          │
│                    │       ┌────┴────┐       │                          │
│                    │       │         │       │                          │
│                    ▼       ▼         ▼       ▼                          │
│                [...]   [B1]       [B2]    [...]                         │
│                       score: 0.8  score: 0.95                           │
│                                      │                                   │
│                            ┌─────────┴─────────┐                        │
│                            │                   │                        │
│                            ▼                   ▼                        │
│                         [B2a]              [B2b]                        │
│                       score: 0.92        score: 0.99 ← BEST PATH        │
│                                              │                          │
│                                              ▼                          │
│                                          [SOLUTION]                     │
│                                                                          │
│   Key: Uses Monte Carlo Tree Search (MCTS) to explore action space     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Architecture Comparison

| Architecture | Best For | Complexity | Reliability |
|--------------|----------|------------|-------------|
| **ReAct** | General tasks, Q&A | Low | Medium |
| **Plan-Execute** | Multi-step projects | Medium | High |
| **Reflexion** | Tasks needing iteration | Medium | High |
| **LATS** | Complex decision-making | High | Very High |

## Choosing the Right Architecture

```yaml
use_react_when:
  - "Task is straightforward"
  - "Real-time response needed"
  - "Few steps required"
  - "Debugging/interpretability important"

use_plan_execute_when:
  - "Task has multiple phases"
  - "Steps have dependencies"
  - "Task may take time"
  - "Need to track progress"

use_reflexion_when:
  - "Task has clear success criteria"
  - "Mistakes are acceptable"
  - "Learning from failures valuable"
  - "Multiple attempts allowed"

use_lats_when:
  - "Many possible approaches"
  - "Optimal solution required"
  - "Cost of mistakes is high"
  - "Time for exploration available"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  Agent Architectures - Summary                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ReAct:         Thought → Action → Observation → Repeat                 │
│                 Simple, interpretable, good baseline                    │
│                                                                          │
│  Plan-Execute:  Plan first → Execute steps → Replan if needed          │
│                 Better for complex, multi-step tasks                    │
│                                                                          │
│  Reflexion:     Try → Evaluate → Reflect → Retry with learning         │
│                 Self-improving through failure analysis                 │
│                                                                          │
│  LATS:          Explore tree of possibilities → Select best path       │
│                 Most thorough, highest quality, slowest                 │
│                                                                          │
│  Common Pattern: All combine LLM reasoning with tool execution          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Agent Components](/learn/agents/agent-fundamentals/components) →
