# Defining Agent Roles

Learn how to design and implement specialized roles for agents in multi-agent systems.

## Why Define Roles?

Clear role definitions lead to more effective multi-agent systems with better specialization, coordination, and quality outcomes.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Benefits of Role-Based Design                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Clear Responsibilities  →  Reduced confusion, better accountability   │
│  Specialization          →  Better quality output per task              │
│  Resource Efficiency     →  Use right tool for each task                │
│  Scalability             →  Easy to add more agents with same role      │
│  Error Recovery          →  Isolate failures, retry at role level       │
│  Auditability            →  Track which agent did what                  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Example: Document Analysis Workflow

```yaml
Generic Approach (1 Agent):
  Agent: "Read document and produce analysis"
  Quality: Shallow, surface-level
  Speed: Slow (does everything sequentially in one head)
  Cost: High (overqualified model handles all tasks)

Role-Based Approach (3 Agents):
  Researcher: Deep content analysis (uses search, extraction)
  Analyst: Synthesize findings into insights
  Reviewer: Quality check, factual verification
  
  Quality: Deep, nuanced, verified
  Speed: Fast (parallel where possible, specialized focus)
  Cost: Optimized (use right model size for each role)
```

## Role Definition Framework

```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional
from enum import Enum

class RoleType(Enum):
    """Types of roles agents can fill."""
    RESEARCHER = "researcher"
    WRITER = "writer"
    REVIEWER = "reviewer"
    CODER = "coder"
    ANALYST = "analyst"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CUSTOM = "custom"

@dataclass
class RoleDefinition:
    """Complete definition of an agent role."""
    
    name: str
    role_type: RoleType
    description: str
    responsibilities: List[str]
    capabilities: List[str]
    limitations: List[str]
    system_prompt: str
    tools: List[str] = field(default_factory=list)
    can_delegate_to: List[str] = field(default_factory=list)
    reports_to: Optional[str] = None
    
    def to_system_message(self) -> str:
        """Convert role to system message."""
        
        return f"""You are a {self.name}.

{self.description}

Your responsibilities:
{chr(10).join(f"- {r}" for r in self.responsibilities)}

Your capabilities:
{chr(10).join(f"- {c}" for c in self.capabilities)}

Limitations (do NOT do these):
{chr(10).join(f"- {l}" for l in self.limitations)}

{self.system_prompt}"""


class RoleRegistry:
    """Registry of all available roles."""
    
    def __init__(self):
        self.roles: Dict[str, RoleDefinition] = {}
        self._setup_default_roles()
    
    def _setup_default_roles(self):
        """Set up standard roles."""
        
        self.register(RoleDefinition(
            name="Research Specialist",
            role_type=RoleType.RESEARCHER,
            description="Expert at gathering and synthesizing information.",
            responsibilities=[
                "Gather comprehensive information on topics",
                "Verify facts from multiple sources",
                "Synthesize findings into clear summaries",
                "Identify gaps in available information"
            ],
            capabilities=[
                "Deep research on any topic",
                "Fact verification",
                "Source evaluation",
                "Information synthesis"
            ],
            limitations=[
                "Do not make up information",
                "Do not write final content (only research)",
                "Do not make decisions outside research scope"
            ],
            system_prompt="Focus on accuracy and completeness. Always cite your reasoning.",
            tools=["web_search", "document_reader"]
        ))
        
        self.register(RoleDefinition(
            name="Content Writer",
            role_type=RoleType.WRITER,
            description="Skilled at creating clear, engaging content.",
            responsibilities=[
                "Write clear, engaging content",
                "Follow style guidelines",
                "Incorporate research findings",
                "Structure content logically"
            ],
            capabilities=[
                "Technical writing",
                "Creative writing",
                "Adapting tone and style",
                "Content structuring"
            ],
            limitations=[
                "Do not conduct research (use provided research)",
                "Do not publish without review",
                "Do not deviate from approved outlines"
            ],
            system_prompt="Write with clarity and engagement. Make complex topics accessible.",
            tools=["text_editor", "grammar_checker"]
        ))
        
        self.register(RoleDefinition(
            name="Quality Reviewer",
            role_type=RoleType.REVIEWER,
            description="Critical eye for quality and accuracy.",
            responsibilities=[
                "Review content for accuracy",
                "Check for clarity and readability",
                "Identify errors and inconsistencies",
                "Provide constructive feedback"
            ],
            capabilities=[
                "Critical analysis",
                "Error detection",
                "Constructive feedback",
                "Quality assessment"
            ],
            limitations=[
                "Do not rewrite content (only suggest changes)",
                "Do not approve your own work",
                "Do not skip review steps"
            ],
            system_prompt="Be thorough but constructive. Provide specific, actionable feedback.",
            tools=["diff_viewer", "checklist_validator"]
        ))
    
    def register(self, role: RoleDefinition):
        """Register a role."""
        self.roles[role.name] = role
    
    def get(self, name: str) -> Optional[RoleDefinition]:
        """Get a role by name."""
        return self.roles.get(name)
    
    def list_roles(self) -> List[str]:
        """List all registered roles."""
        return list(self.roles.keys())
```

## Creating Role-Based Agents

```python
from openai import AsyncOpenAI
from typing import Any

class RoleBasedAgent:
    """Agent that operates according to a defined role."""
    
    def __init__(
        self,
        agent_id: str,
        role: RoleDefinition,
        llm_client: AsyncOpenAI,
        model: str = "gpt-4"
    ):
        self.id = agent_id
        self.role = role
        self.llm = llm_client
        self.model = model
        self.conversation_history: List[Dict[str, str]] = []
    
    async def execute(self, task: str, context: Dict[str, Any] = None) -> str:
        """Execute a task within role constraints."""
        
        # Build messages
        messages = [
            {"role": "system", "content": self.role.to_system_message()}
        ]
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        # Build task prompt with context
        task_prompt = f"Task: {task}"
        
        if context:
            context_str = "\n".join(
                f"{k}: {v}" for k, v in context.items()
            )
            task_prompt += f"\n\nContext:\n{context_str}"
        
        messages.append({"role": "user", "content": task_prompt})
        
        # Execute
        response = await self.llm.chat.completions.create(
            model=self.model,
            messages=messages
        )
        
        result = response.choices[0].message.content
        
        # Update history
        self.conversation_history.append({"role": "user", "content": task_prompt})
        self.conversation_history.append({"role": "assistant", "content": result})
        
        return result
    
    def can_handle(self, task_type: str) -> bool:
        """Check if this agent can handle a task type."""
        return task_type.lower() in [c.lower() for c in self.role.capabilities]
    
    def get_available_tools(self) -> List[str]:
        """Get tools this agent can use."""
        return self.role.tools


class AgentFactory:
    """Factory for creating role-based agents."""
    
    def __init__(self, llm_client: AsyncOpenAI, role_registry: RoleRegistry):
        self.llm = llm_client
        self.registry = role_registry
        self.agents: Dict[str, RoleBasedAgent] = {}
    
    def create(
        self,
        agent_id: str,
        role_name: str,
        model: str = "gpt-4"
    ) -> RoleBasedAgent:
        """Create an agent with a specific role."""
        
        role = self.registry.get(role_name)
        if not role:
            raise ValueError(f"Unknown role: {role_name}")
        
        agent = RoleBasedAgent(
            agent_id=agent_id,
            role=role,
            llm_client=self.llm,
            model=model
        )
        
        self.agents[agent_id] = agent
        return agent
    
    def create_custom(
        self,
        agent_id: str,
        role_definition: RoleDefinition,
        model: str = "gpt-4"
    ) -> RoleBasedAgent:
        """Create an agent with a custom role."""
        
        agent = RoleBasedAgent(
            agent_id=agent_id,
            role=role_definition,
            llm_client=self.llm,
            model=model
        )
        
        self.agents[agent_id] = agent
        return agent
    
    def get(self, agent_id: str) -> Optional[RoleBasedAgent]:
        """Get an agent by ID."""
        return self.agents.get(agent_id)
```

## Common Role Patterns

### Pattern 1: Research Team

```python
def create_research_team(factory: AgentFactory) -> Dict[str, RoleBasedAgent]:
    """Create a research-focused team."""
    
    # Custom roles for research
    lead_researcher = RoleDefinition(
        name="Lead Researcher",
        role_type=RoleType.RESEARCHER,
        description="Leads research efforts and synthesizes findings.",
        responsibilities=[
            "Define research questions",
            "Coordinate research activities",
            "Synthesize team findings",
            "Identify knowledge gaps"
        ],
        capabilities=["Research planning", "Synthesis", "Team coordination"],
        limitations=["Don't do detailed research (delegate it)"],
        system_prompt="Think strategically about research direction.",
        can_delegate_to=["Data Researcher", "Literature Researcher"]
    )
    
    data_researcher = RoleDefinition(
        name="Data Researcher",
        role_type=RoleType.RESEARCHER,
        description="Specializes in data and statistics.",
        responsibilities=[
            "Find relevant data and statistics",
            "Verify numerical claims",
            "Analyze data trends"
        ],
        capabilities=["Data analysis", "Statistical research", "Chart interpretation"],
        limitations=["Don't analyze qualitative content"],
        system_prompt="Focus on quantitative information.",
        reports_to="Lead Researcher"
    )
    
    lit_researcher = RoleDefinition(
        name="Literature Researcher",
        role_type=RoleType.RESEARCHER,
        description="Specializes in academic and written sources.",
        responsibilities=[
            "Review academic literature",
            "Summarize key papers",
            "Track citations and sources"
        ],
        capabilities=["Literature review", "Academic research", "Citation tracking"],
        limitations=["Don't analyze raw data"],
        system_prompt="Focus on published research and expert opinions.",
        reports_to="Lead Researcher"
    )
    
    return {
        "lead": factory.create_custom("research_lead", lead_researcher),
        "data": factory.create_custom("data_researcher", data_researcher),
        "literature": factory.create_custom("lit_researcher", lit_researcher)
    }
```

### Pattern 2: Content Team

```python
def create_content_team(factory: AgentFactory) -> Dict[str, RoleBasedAgent]:
    """Create a content creation team."""
    
    editor_role = RoleDefinition(
        name="Editor in Chief",
        role_type=RoleType.PLANNER,
        description="Oversees content strategy and quality.",
        responsibilities=[
            "Define content direction",
            "Assign writing tasks",
            "Final approval of content",
            "Ensure consistency"
        ],
        capabilities=["Content strategy", "Editorial oversight", "Quality control"],
        limitations=["Don't write content directly"],
        system_prompt="Focus on strategic direction and quality standards.",
        can_delegate_to=["Technical Writer", "Creative Writer", "Copy Editor"]
    )
    
    tech_writer = RoleDefinition(
        name="Technical Writer",
        role_type=RoleType.WRITER,
        description="Writes technical and educational content.",
        responsibilities=[
            "Write clear technical explanations",
            "Create tutorials and guides",
            "Document processes"
        ],
        capabilities=["Technical writing", "Tutorial creation", "Documentation"],
        limitations=["Don't write marketing copy", "Don't include humor"],
        system_prompt="Write clearly and precisely. Assume the reader is smart but unfamiliar with the topic.",
        reports_to="Editor in Chief"
    )
    
    creative_writer = RoleDefinition(
        name="Creative Writer",
        role_type=RoleType.WRITER,
        description="Writes engaging, creative content.",
        responsibilities=[
            "Write engaging narratives",
            "Create compelling headlines",
            "Add personality to content"
        ],
        capabilities=["Creative writing", "Storytelling", "Engagement"],
        limitations=["Don't write dry technical docs"],
        system_prompt="Write with personality and engagement. Make readers want to continue.",
        reports_to="Editor in Chief"
    )
    
    copy_editor = RoleDefinition(
        name="Copy Editor",
        role_type=RoleType.REVIEWER,
        description="Polishes and corrects written content.",
        responsibilities=[
            "Fix grammar and spelling",
            "Improve sentence structure",
            "Ensure style consistency"
        ],
        capabilities=["Proofreading", "Style editing", "Grammar correction"],
        limitations=["Don't change meaning", "Don't rewrite substantially"],
        system_prompt="Focus on polish and consistency. Preserve the author's voice.",
        reports_to="Editor in Chief"
    )
    
    return {
        "editor": factory.create_custom("editor_chief", editor_role),
        "technical": factory.create_custom("tech_writer", tech_writer),
        "creative": factory.create_custom("creative_writer", creative_writer),
        "copy": factory.create_custom("copy_editor", copy_editor)
    }
```

### Pattern 3: Development Team

```python
def create_dev_team(factory: AgentFactory) -> Dict[str, RoleBasedAgent]:
    """Create a software development team."""
    
    architect = RoleDefinition(
        name="Software Architect",
        role_type=RoleType.PLANNER,
        description="Designs system architecture and patterns.",
        responsibilities=[
            "Design system architecture",
            "Define coding standards",
            "Review technical decisions",
            "Guide implementation approach"
        ],
        capabilities=["System design", "Architecture patterns", "Technical leadership"],
        limitations=["Don't write implementation code", "Don't do testing"],
        system_prompt="Think about scalability, maintainability, and best practices.",
        can_delegate_to=["Backend Developer", "Frontend Developer", "QA Engineer"]
    )
    
    backend_dev = RoleDefinition(
        name="Backend Developer",
        role_type=RoleType.CODER,
        description="Implements server-side logic.",
        responsibilities=[
            "Write backend code",
            "Implement APIs",
            "Handle database operations",
            "Write unit tests"
        ],
        capabilities=["Python", "APIs", "Databases", "Server logic"],
        limitations=["Don't write frontend code", "Don't make architecture decisions alone"],
        system_prompt="Write clean, tested, efficient backend code.",
        tools=["code_executor", "database_client"],
        reports_to="Software Architect"
    )
    
    frontend_dev = RoleDefinition(
        name="Frontend Developer",
        role_type=RoleType.CODER,
        description="Implements user interfaces.",
        responsibilities=[
            "Write frontend code",
            "Implement UI components",
            "Handle user interactions",
            "Ensure accessibility"
        ],
        capabilities=["JavaScript", "React", "CSS", "UI/UX"],
        limitations=["Don't write backend code", "Don't modify APIs"],
        system_prompt="Write clean, accessible, responsive frontend code.",
        tools=["code_executor", "browser_preview"],
        reports_to="Software Architect"
    )
    
    qa_engineer = RoleDefinition(
        name="QA Engineer",
        role_type=RoleType.REVIEWER,
        description="Tests and validates software quality.",
        responsibilities=[
            "Write test cases",
            "Execute tests",
            "Report bugs",
            "Verify fixes"
        ],
        capabilities=["Testing", "Bug detection", "Test automation"],
        limitations=["Don't fix bugs directly", "Don't approve own tests"],
        system_prompt="Be thorough in testing. Think about edge cases.",
        tools=["test_runner", "bug_tracker"],
        reports_to="Software Architect"
    )
    
    return {
        "architect": factory.create_custom("sw_architect", architect),
        "backend": factory.create_custom("backend_dev", backend_dev),
        "frontend": factory.create_custom("frontend_dev", frontend_dev),
        "qa": factory.create_custom("qa_engineer", qa_engineer)
    }
```

## Dynamic Role Assignment

```python
class DynamicRoleAssigner:
    """Dynamically assign roles based on task requirements."""
    
    def __init__(self, llm_client: AsyncOpenAI, role_registry: RoleRegistry):
        self.llm = llm_client
        self.registry = role_registry
    
    async def assign_roles(self, task: str) -> List[str]:
        """Determine which roles are needed for a task."""
        
        available_roles = "\n".join(
            f"- {name}: {role.description}"
            for name, role in self.registry.roles.items()
        )
        
        prompt = f"""Analyze this task and determine which roles are needed:

Task: {task}

Available Roles:
{available_roles}

Return a JSON list of role names needed for this task, in order of involvement:
{{"roles": ["Role Name 1", "Role Name 2", ...]}}
"""
        
        response = await self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        
        import json
        data = json.loads(response.choices[0].message.content)
        
        return data["roles"]
    
    async def create_team_for_task(
        self,
        task: str,
        factory: AgentFactory
    ) -> Dict[str, RoleBasedAgent]:
        """Create a team of agents for a specific task."""
        
        needed_roles = await self.assign_roles(task)
        
        team = {}
        for i, role_name in enumerate(needed_roles):
            agent_id = f"agent_{role_name.lower().replace(' ', '_')}_{i}"
            team[role_name] = factory.create(agent_id, role_name)
        
        return team
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Agent Roles - Summary                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Role Definition Components:                                             │
│    • Name and type                                                      │
│    • Description and purpose                                            │
│    • Responsibilities (what they DO)                                    │
│    • Capabilities (what they CAN do)                                    │
│    • Limitations (what they DON'T do)                                   │
│    • System prompt for LLM                                              │
│    • Tools and access                                                   │
│                                                                          │
│  Common Team Patterns:                                                   │
│    • Research Team: Lead + specialists                                  │
│    • Content Team: Editor + writers + reviewers                        │
│    • Dev Team: Architect + developers + QA                             │
│                                                                          │
│  Benefits of Clear Roles:                                                │
│    • Better specialization                                              │
│    • Clearer accountability                                             │
│    • Reduced confusion/overlap                                         │
│    • More consistent outputs                                           │
│                                                                          │
│  Best Practices:                                                         │
│    • Define clear boundaries                                           │
│    • Specify what NOT to do                                            │
│    • Include delegation paths                                          │
│    • Use role-specific prompts                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Message Passing](/learn/multi-agents/building-multi-agent/message-passing) →
