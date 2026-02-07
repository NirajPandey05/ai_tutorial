# What are AI Agents?

Understand the fundamental concept of AI agents and how they differ from traditional chatbots.

## Defining AI Agents

An **AI Agent** is an autonomous system that can perceive its environment, reason about it, and take actions to achieve specific goals—all powered by Large Language Models.

```yaml
agent_definition:
  core_concept: "An LLM-powered system that can autonomously take actions"
  key_traits:
    - "Autonomous decision-making"
    - "Goal-oriented behavior"
    - "Tool usage capabilities"
    - "Iterative problem-solving"
  analogy: "A skilled assistant who can think, plan, and execute tasks independently"
```

## Agent vs Chatbot

Understanding the difference is crucial:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Chatbot vs AI Agent                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  CHATBOT                           │  AI AGENT                          │
│  ────────                          │  ─────────                         │
│                                    │                                    │
│  • Responds to messages            │  • Takes autonomous actions        │
│  • Single turn interactions        │  • Multi-step task execution       │
│  • No external tool access         │  • Uses tools and APIs             │
│  • Stateless conversations         │  • Maintains memory and context    │
│  • Pre-defined responses           │  • Dynamic problem-solving         │
│  • Human initiates each step       │  • Self-directed execution         │
│                                    │                                    │
│  User: "What's the weather?"       │  User: "Book me a flight to NYC"   │
│  Bot: "I can't check weather."     │  Agent: *searches flights*         │
│                                    │         *compares prices*          │
│  ───────────────────────────────   │         *selects best option*      │
│                                    │         *fills booking form*       │
│                                    │         *confirms with user*       │
│                                    │                                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Differences Table

| Aspect | Chatbot | AI Agent |
|--------|---------|----------|
| **Interaction** | Reactive | Proactive |
| **Scope** | Single response | Multi-step workflows |
| **Tools** | None or limited | Extensive tool access |
| **Memory** | Session-based | Persistent memory |
| **Autonomy** | Low | High |
| **Decision-making** | Rule-based | Reasoning-based |

## Core Capabilities of AI Agents

### 1. Reasoning

Agents can think through problems step-by-step:

```python
# Example: Agent reasoning about a task
"""
User Request: "Find the best restaurant for a team dinner"

Agent Reasoning:
1. I need to find restaurants near the user's location
2. I should consider: capacity, cuisine preferences, budget
3. Let me search for restaurants with good reviews
4. I'll filter by availability for the team size
5. Present top 3 options with pros/cons
"""
```

### 2. Planning

Breaking complex tasks into manageable steps:

```python
# Example: Agent planning a research task
task = "Write a report on AI trends in 2026"

plan = [
    "1. Search for recent AI news and publications",
    "2. Identify top 5 trending topics",
    "3. Gather statistics and data points",
    "4. Outline the report structure",
    "5. Write each section with citations",
    "6. Review and refine the content",
    "7. Format and present the final report"
]
```

### 3. Tool Use

Interacting with external systems:

```python
# Example: Agent using tools
available_tools = {
    "web_search": "Search the internet for information",
    "calculator": "Perform mathematical calculations",
    "code_executor": "Run Python code",
    "file_manager": "Read and write files",
    "api_caller": "Make HTTP API requests",
    "database": "Query databases"
}

# Agent decides which tool to use based on the task
# "What's 15% tip on $84.50?" → calculator
# "Latest news about SpaceX" → web_search
# "Create a bar chart of sales data" → code_executor
```

### 4. Memory

Maintaining context across interactions:

```python
# Example: Agent memory types
class AgentMemory:
    def __init__(self):
        # Short-term: Current conversation
        self.working_memory = []
        
        # Long-term: Persistent knowledge
        self.long_term_memory = VectorStore()
        
        # Episodic: Past experiences
        self.episode_memory = []
        
        # Semantic: Facts and relationships
        self.knowledge_base = {}
```

## The Agent Spectrum

Agents exist on a spectrum of autonomy:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Agent Autonomy Spectrum                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   LOW AUTONOMY ◄──────────────────────────────────────► HIGH AUTONOMY   │
│                                                                          │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐   │
│   │  Chatbot    │  │  Assistant  │  │  Copilot    │  │  Autonomous │   │
│   │             │  │             │  │             │  │   Agent     │   │
│   │ • Q&A only  │  │ • Suggests  │  │ • Executes  │  │ • Plans &   │   │
│   │ • No tools  │  │   actions   │  │   with      │  │   executes  │   │
│   │ • No memory │  │ • Limited   │  │   approval  │  │ • Full tool │   │
│   │             │  │   tools     │  │ • Some      │  │   access    │   │
│   │             │  │             │  │   autonomy  │  │ • Self-     │   │
│   │             │  │             │  │             │  │   directed  │   │
│   └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘   │
│                                                                          │
│   Example:         Example:         Example:         Example:           │
│   FAQ Bot          Siri/Alexa       GitHub Copilot   AutoGPT            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Real-World Agent Examples

### 1. Customer Service Agent
```python
class CustomerServiceAgent:
    """Agent that handles customer inquiries autonomously."""
    
    tools = [
        "lookup_order",      # Check order status
        "process_refund",    # Issue refunds
        "update_address",    # Modify shipping info
        "escalate_ticket",   # Escalate to human
        "send_email",        # Send confirmation emails
    ]
    
    async def handle_inquiry(self, message: str):
        # 1. Understand the request
        intent = await self.classify_intent(message)
        
        # 2. Gather necessary information
        context = await self.get_customer_context()
        
        # 3. Execute appropriate actions
        if intent == "refund_request":
            order = await self.lookup_order(context.order_id)
            if self.is_refund_eligible(order):
                await self.process_refund(order)
                await self.send_email("refund_confirmation")
        
        # 4. Respond to customer
        return self.generate_response()
```

### 2. Research Agent
```python
class ResearchAgent:
    """Agent that conducts research and writes reports."""
    
    tools = [
        "web_search",        # Search the internet
        "read_webpage",      # Extract content from URLs
        "summarize",         # Summarize long texts
        "write_document",    # Create documents
        "cite_sources",      # Add citations
    ]
    
    async def research_topic(self, topic: str):
        # 1. Plan the research
        subtopics = await self.break_down_topic(topic)
        
        # 2. Gather information
        sources = []
        for subtopic in subtopics:
            results = await self.web_search(subtopic)
            for result in results:
                content = await self.read_webpage(result.url)
                sources.append({
                    "content": content,
                    "url": result.url
                })
        
        # 3. Synthesize findings
        report = await self.write_document(
            topic=topic,
            sources=sources,
            style="academic"
        )
        
        return report
```

### 3. Coding Agent
```python
class CodingAgent:
    """Agent that writes and debugs code."""
    
    tools = [
        "read_file",         # Read source files
        "write_file",        # Write/modify code
        "run_code",          # Execute code
        "search_docs",       # Search documentation
        "run_tests",         # Run test suite
    ]
    
    async def implement_feature(self, specification: str):
        # 1. Understand the codebase
        structure = await self.analyze_codebase()
        
        # 2. Plan the implementation
        plan = await self.create_implementation_plan(specification)
        
        # 3. Write the code
        for step in plan:
            code = await self.generate_code(step)
            await self.write_file(step.file_path, code)
        
        # 4. Test and iterate
        while True:
            results = await self.run_tests()
            if results.all_passed:
                break
            await self.fix_failing_tests(results)
        
        return "Feature implemented successfully"
```

## When to Use Agents

```yaml
good_use_cases:
  - "Complex multi-step tasks"
  - "Tasks requiring external data/tools"
  - "Processes that benefit from iteration"
  - "Workflows needing dynamic decisions"
  - "Tasks with clear success criteria"

poor_use_cases:
  - "Simple Q&A (use chatbot)"
  - "Real-time responses needed"
  - "High-stakes decisions without oversight"
  - "Tasks with no clear completion criteria"
  - "Simple CRUD operations"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    What are AI Agents - Summary                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Definition:                                                             │
│    LLM-powered systems that autonomously reason and act                 │
│                                                                          │
│  Core Capabilities:                                                      │
│    • Reasoning: Think through problems step-by-step                     │
│    • Planning: Break complex tasks into steps                           │
│    • Tool Use: Interact with external systems                           │
│    • Memory: Maintain context across interactions                       │
│                                                                          │
│  Key Difference from Chatbots:                                          │
│    Agents TAKE ACTIONS, chatbots just RESPOND                           │
│                                                                          │
│  Autonomy Levels:                                                        │
│    Chatbot → Assistant → Copilot → Autonomous Agent                     │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Agent Architectures](/learn/agents/agent-fundamentals/architectures) →
