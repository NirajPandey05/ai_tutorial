# Prompts in MCP

Prompts are reusable prompt templates that MCP servers can expose to clients, enabling consistent and parameterized interactions with LLMs.

## What are Prompts?

Prompts in MCP are **pre-defined templates** that servers provide to help users interact with LLMs more effectively. They encapsulate best practices, domain knowledge, and structured workflows.

```yaml
prompts_overview:
  definition: "Reusable prompt templates with optional parameters"
  
  characteristics:
    - "Parameterized with arguments"
    - "Can include multiple messages"
    - "Can embed resources"
    - "Server-provided best practices"
  
  examples:
    - "Code review prompt with file input"
    - "SQL query generator with schema context"
    - "Document summarizer with style options"
    - "Debug helper with error context"
```

```
┌─────────────────────────────────────────────────────────────────┐
│                How Prompts Differ from Tools                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   TOOLS                           PROMPTS                        │
│   ─────                           ───────                        │
│   • Execute actions               • Generate LLM input          │
│   • Server-side execution         • Client-side expansion       │
│   • Return results                • Return message templates    │
│   • "Do something"                • "Say something"             │
│                                                                  │
│   Example:                        Example:                       │
│   "Search the database"           "Here's how to analyze this   │
│   → Returns results               code: [template with context]"│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Prompt Structure

Each prompt has a name, description, optional arguments, and returns messages:

```python
from mcp.server import Server
from mcp.types import Prompt, PromptArgument, PromptMessage, TextContent

server = Server("prompt-demo")

@server.list_prompts()
async def list_prompts():
    """List available prompts."""
    return [
        Prompt(
            name="code_review",
            description="Generate a thorough code review",
            arguments=[
                PromptArgument(
                    name="code",
                    description="The code to review",
                    required=True
                ),
                PromptArgument(
                    name="language",
                    description="Programming language",
                    required=False
                ),
                PromptArgument(
                    name="focus",
                    description="Review focus (security, performance, style)",
                    required=False
                )
            ]
        )
    ]
```

## Implementing Prompts

### Basic Prompt Implementation

```python
@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    """Get a specific prompt with arguments filled in."""
    
    if name == "code_review":
        code = arguments.get("code", "") if arguments else ""
        language = arguments.get("language", "unknown") if arguments else "unknown"
        focus = arguments.get("focus", "general") if arguments else "general"
        
        return {
            "description": f"Code review for {language} code",
            "messages": [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Please review the following {language} code with a focus on {focus}.

Provide feedback on:
1. Code correctness and potential bugs
2. {focus.capitalize()} considerations
3. Code readability and maintainability
4. Suggestions for improvement

Code to review:
```{language}
{code}
```

Format your response as:
- 🐛 Bugs/Issues
- ⚠️ Warnings
- 💡 Suggestions
- ✅ Good practices noted"""
                    )
                )
            ]
        }
    
    raise ValueError(f"Unknown prompt: {name}")
```

### Multi-Message Prompts

```python
@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    if name == "sql_expert":
        schema = arguments.get("schema", "") if arguments else ""
        question = arguments.get("question", "") if arguments else ""
        
        return {
            "description": "SQL query generation with schema context",
            "messages": [
                # System message with context
                PromptMessage(
                    role="system",
                    content=TextContent(
                        type="text",
                        text=f"""You are an expert SQL developer. You have access to a database with the following schema:

{schema}

Guidelines:
- Write efficient, readable SQL
- Use proper JOINs (avoid subqueries when possible)
- Add comments explaining complex logic
- Consider index usage for performance
- Always use parameterized queries format (use ? placeholders)"""
                    )
                ),
                # User message with the question
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Write a SQL query to answer this question:

{question}

Provide:
1. The SQL query
2. Explanation of the query logic
3. Any performance considerations"""
                    )
                )
            ]
        }
```

### Prompts with Embedded Resources

```python
@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    if name == "analyze_project":
        project_path = arguments.get("path", ".") if arguments else "."
        
        # Embed resources for context
        return {
            "description": f"Analyze project at {project_path}",
            "messages": [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text="""Please analyze this project and provide:

1. **Overview**: What does this project do?
2. **Architecture**: Key components and how they interact
3. **Dependencies**: Main dependencies and their purposes
4. **Suggestions**: Areas for improvement

Here are the relevant project files:"""
                    )
                ),
                # Embed README resource
                PromptMessage(
                    role="user",
                    content=EmbeddedResource(
                        type="resource",
                        resource=Resource(
                            uri=f"file://{project_path}/README.md",
                            name="README",
                            mimeType="text/markdown"
                        )
                    )
                ),
                # Embed package.json or pyproject.toml
                PromptMessage(
                    role="user", 
                    content=EmbeddedResource(
                        type="resource",
                        resource=Resource(
                            uri=f"file://{project_path}/package.json",
                            name="Dependencies",
                            mimeType="application/json"
                        )
                    )
                )
            ]
        }
```

## Prompt Argument Types

### Required vs Optional Arguments

```python
Prompt(
    name="document_summary",
    description="Summarize a document with customization options",
    arguments=[
        # Required: Must be provided
        PromptArgument(
            name="document",
            description="The document text to summarize",
            required=True
        ),
        # Optional: Has sensible defaults
        PromptArgument(
            name="length",
            description="Summary length: brief, standard, detailed",
            required=False
        ),
        PromptArgument(
            name="style",
            description="Writing style: formal, casual, technical",
            required=False
        ),
        PromptArgument(
            name="audience",
            description="Target audience for the summary",
            required=False
        )
    ]
)
```

### Argument Validation in Implementation

```python
@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    if name == "document_summary":
        # Extract with defaults
        document = arguments.get("document") if arguments else None
        if not document:
            raise ValueError("Document is required")
        
        length = arguments.get("length", "standard") if arguments else "standard"
        style = arguments.get("style", "formal") if arguments else "formal"
        audience = arguments.get("audience", "general") if arguments else "general"
        
        # Validate enum-like values
        valid_lengths = ["brief", "standard", "detailed"]
        if length not in valid_lengths:
            length = "standard"
        
        length_instructions = {
            "brief": "2-3 sentences maximum",
            "standard": "1-2 paragraphs",
            "detailed": "comprehensive multi-paragraph summary"
        }
        
        return {
            "description": f"{length.capitalize()} {style} summary",
            "messages": [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Summarize the following document.

Requirements:
- Length: {length_instructions[length]}
- Style: {style}
- Target audience: {audience}

Document:
{document}"""
                    )
                )
            ]
        }
```

## Real-World Prompt Examples

### Code Generation Prompt

```python
Prompt(
    name="generate_function",
    description="Generate a function based on requirements",
    arguments=[
        PromptArgument(name="description", description="What the function should do", required=True),
        PromptArgument(name="language", description="Programming language", required=True),
        PromptArgument(name="inputs", description="Input parameters description", required=False),
        PromptArgument(name="outputs", description="Expected return value", required=False),
        PromptArgument(name="examples", description="Example usage", required=False)
    ]
)

@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    if name == "generate_function":
        desc = arguments.get("description", "")
        lang = arguments.get("language", "python")
        inputs = arguments.get("inputs", "Not specified")
        outputs = arguments.get("outputs", "Not specified")
        examples = arguments.get("examples", "")
        
        return {
            "messages": [
                PromptMessage(
                    role="system",
                    content=TextContent(
                        type="text",
                        text=f"""You are an expert {lang} developer. Generate clean, well-documented code.

Include:
- Clear function signature with type hints
- Docstring with description, parameters, returns
- Error handling where appropriate
- Example usage in comments"""
                    )
                ),
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Generate a {lang} function with these requirements:

**Description**: {desc}

**Inputs**: {inputs}

**Outputs**: {outputs}

{f'**Examples**: {examples}' if examples else ''}

Provide the complete implementation with documentation."""
                    )
                )
            ]
        }
```

### Debug Helper Prompt

```python
Prompt(
    name="debug_error",
    description="Help debug an error or exception",
    arguments=[
        PromptArgument(name="error_message", description="The error message/traceback", required=True),
        PromptArgument(name="code_context", description="Relevant code that caused the error", required=False),
        PromptArgument(name="language", description="Programming language", required=False),
        PromptArgument(name="what_i_tried", description="What you've already tried", required=False)
    ]
)

@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    if name == "debug_error":
        error = arguments.get("error_message", "")
        code = arguments.get("code_context", "")
        lang = arguments.get("language", "")
        tried = arguments.get("what_i_tried", "")
        
        return {
            "messages": [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""I'm encountering an error and need help debugging it.

**Error Message:**
```
{error}
```

{f'**Code Context ({lang}):**' if code else ''}
{f'```{lang}' if code else ''}
{code if code else ''}
{f'```' if code else ''}

{f'**What I have tried:** {tried}' if tried else ''}

Please help me:
1. **Understand** what this error means
2. **Identify** the likely cause
3. **Suggest** how to fix it
4. **Explain** how to prevent similar errors"""
                    )
                )
            ]
        }
```

### Documentation Generator Prompt

```python
Prompt(
    name="generate_docs",
    description="Generate documentation for code",
    arguments=[
        PromptArgument(name="code", description="Code to document", required=True),
        PromptArgument(name="doc_type", description="Type: api, readme, tutorial, inline", required=False),
        PromptArgument(name="format", description="Format: markdown, rst, jsdoc", required=False)
    ]
)

@server.get_prompt()
async def get_prompt(name: str, arguments: dict | None = None):
    if name == "generate_docs":
        code = arguments.get("code", "")
        doc_type = arguments.get("doc_type", "api")
        format = arguments.get("format", "markdown")
        
        type_instructions = {
            "api": "Generate API reference documentation with function signatures, parameters, return values, and examples",
            "readme": "Generate a README with overview, installation, usage examples, and API summary",
            "tutorial": "Generate a step-by-step tutorial explaining how to use this code",
            "inline": "Add inline comments and docstrings to the code"
        }
        
        return {
            "messages": [
                PromptMessage(
                    role="user",
                    content=TextContent(
                        type="text",
                        text=f"""Generate {doc_type} documentation for the following code.

**Documentation Type**: {type_instructions.get(doc_type, type_instructions['api'])}
**Format**: {format}

**Code:**
```
{code}
```

Make the documentation:
- Clear and comprehensive
- Include practical examples
- Follow {format} best practices"""
                    )
                )
            ]
        }
```

## Client Usage

### Getting and Using Prompts

```python
from mcp import ClientSession

async def use_prompts(session: ClientSession):
    # List available prompts
    prompts = await session.list_prompts()
    print(f"Available prompts: {[p.name for p in prompts.prompts]}")
    
    # Get a specific prompt
    prompt_result = await session.get_prompt(
        name="code_review",
        arguments={
            "code": "def add(a, b): return a + b",
            "language": "python",
            "focus": "performance"
        }
    )
    
    # Use the messages with your LLM
    messages = [
        {"role": msg.role, "content": msg.content.text}
        for msg in prompt_result.messages
    ]
    
    response = await llm.chat.completions.create(
        model="gpt-4",
        messages=messages
    )
    
    return response
```

### Building a Prompt Selection UI

```python
async def prompt_ui(session: ClientSession):
    """Interactive prompt selection and execution."""
    
    # Fetch available prompts
    prompts = await session.list_prompts()
    
    # Display options
    print("\nAvailable Prompts:")
    for i, prompt in enumerate(prompts.prompts, 1):
        print(f"  {i}. {prompt.name}: {prompt.description}")
        if prompt.arguments:
            for arg in prompt.arguments:
                required = "(required)" if arg.required else "(optional)"
                print(f"      - {arg.name} {required}: {arg.description}")
    
    # User selects prompt
    selection = int(input("\nSelect prompt: ")) - 1
    selected_prompt = prompts.prompts[selection]
    
    # Gather arguments
    arguments = {}
    for arg in selected_prompt.arguments or []:
        value = input(f"  {arg.name}: ")
        if value or arg.required:
            arguments[arg.name] = value
    
    # Get expanded prompt
    result = await session.get_prompt(
        name=selected_prompt.name,
        arguments=arguments
    )
    
    return result.messages
```

## Best Practices

```yaml
prompt_best_practices:
  naming:
    - "Use descriptive, action-oriented names"
    - "Follow consistent naming conventions"
    - "Examples: analyze_code, generate_tests, explain_error"
  
  descriptions:
    - "Clearly explain what the prompt does"
    - "Mention expected output format"
    - "Note any prerequisites or limitations"
  
  arguments:
    - "Make commonly-needed parameters required"
    - "Provide sensible defaults for optional args"
    - "Document valid values for constrained args"
  
  messages:
    - "Use system messages for context/instructions"
    - "Structure user messages clearly"
    - "Include examples when helpful"
  
  content:
    - "Include domain expertise in templates"
    - "Add formatting instructions"
    - "Request structured output when needed"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                      MCP Prompts Summary                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Prompts = Reusable prompt templates with parameters            │
│                                                                  │
│  Components:                                                     │
│    • name: Unique identifier                                    │
│    • description: What the prompt does                          │
│    • arguments: Optional parameters                             │
│    • messages: Template messages returned                       │
│                                                                  │
│  Implementation:                                                 │
│    • list_prompts(): Declare available prompts                  │
│    • get_prompt(): Return filled template                       │
│                                                                  │
│  Message Types:                                                  │
│    • system: Context and instructions                           │
│    • user: The actual query                                     │
│    • assistant: Pre-filled responses (rare)                     │
│                                                                  │
│  Use Cases:                                                      │
│    • Code review templates                                      │
│    • Documentation generators                                   │
│    • Debug helpers                                              │
│    • Domain-specific workflows                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

Next: [MCP Transport Layer](/learn/mcp/mcp-core-concepts/transport) →
