# Function Calling Basics

Learn how to enable LLMs to call functions and interact with external tools.

## What is Function Calling?

Function calling is the mechanism that allows LLMs to request execution of specific functions with structured arguments. It's the foundation of tool use in AI agents.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Function Calling Flow                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│   │  USER   │───►│   LLM   │───►│  YOUR   │───►│   LLM   │──► Response │
│   │ QUERY   │    │ (thinks)│    │  CODE   │    │ (final) │             │
│   └─────────┘    └─────────┘    └─────────┘    └─────────┘             │
│                       │              ▲                                   │
│                       │              │                                   │
│                       ▼              │                                   │
│                 "I need to call      │                                   │
│                  get_weather with    │                                   │
│                  location='Paris'"   │                                   │
│                       │              │                                   │
│                       └──────────────┘                                   │
│                       Function call + result                             │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Basic Function Calling with OpenAI

### Step 1: Define Your Functions

```python
from openai import OpenAI

client = OpenAI()

# Define available tools/functions
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state, e.g., 'San Francisco, CA'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]
```

### Step 2: Make the Initial Request

```python
messages = [
    {"role": "user", "content": "What's the weather like in Paris?"}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"  # Let LLM decide whether to use tools
)

# Check if the model wants to call a function
message = response.choices[0].message

print(f"Stop reason: {response.choices[0].finish_reason}")
print(f"Tool calls: {message.tool_calls}")
```

Output:
```
Stop reason: tool_calls
Tool calls: [
    ChatCompletionMessageToolCall(
        id='call_abc123',
        type='function',
        function=Function(
            name='get_weather',
            arguments='{"location": "Paris, France", "unit": "celsius"}'
        )
    )
]
```

### Step 3: Execute the Function

```python
import json

def get_weather(location: str, unit: str = "celsius") -> dict:
    """Your actual implementation of the weather function."""
    # In reality, you'd call a weather API here
    return {
        "location": location,
        "temperature": 22 if unit == "celsius" else 72,
        "unit": unit,
        "conditions": "Partly cloudy"
    }

# Process tool calls
if message.tool_calls:
    for tool_call in message.tool_calls:
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Execute the function
        if function_name == "get_weather":
            result = get_weather(**arguments)
        
        print(f"Function: {function_name}")
        print(f"Arguments: {arguments}")
        print(f"Result: {result}")
```

### Step 4: Return Result to LLM

```python
# Add assistant's message with tool calls
messages.append(message)

# Add the function result
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})

# Get final response
final_response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools
)

print(final_response.choices[0].message.content)
```

Output:
```
The weather in Paris, France is currently 22°C (72°F) with partly cloudy conditions.
```

## Complete Function Calling Example

```python
from openai import OpenAI
import json
from typing import Dict, Any, Callable

class FunctionCallingAgent:
    """Agent that can call functions based on user queries."""
    
    def __init__(self):
        self.client = OpenAI()
        self.tools = []
        self.functions: Dict[str, Callable] = {}
    
    def register_function(
        self, 
        func: Callable, 
        description: str,
        parameters: Dict[str, Any]
    ):
        """Register a function that the agent can call."""
        
        # Add to tools list for LLM
        self.tools.append({
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": parameters
            }
        })
        
        # Store function reference
        self.functions[func.__name__] = func
    
    def process_query(self, query: str) -> str:
        """Process a user query, potentially calling functions."""
        
        messages = [{"role": "user", "content": query}]
        
        # Make initial request
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools if self.tools else None,
            tool_choice="auto"
        )
        
        message = response.choices[0].message
        
        # If no tool calls, return the response directly
        if not message.tool_calls:
            return message.content
        
        # Process tool calls
        messages.append(message)
        
        for tool_call in message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            # Execute the function
            if function_name in self.functions:
                try:
                    result = self.functions[function_name](**arguments)
                    result_str = json.dumps(result) if not isinstance(result, str) else result
                except Exception as e:
                    result_str = f"Error: {str(e)}"
            else:
                result_str = f"Error: Unknown function {function_name}"
            
            # Add result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result_str
            })
        
        # Get final response
        final_response = self.client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=self.tools
        )
        
        return final_response.choices[0].message.content


# Example usage
def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get the current stock price for a symbol."""
    # Simulated stock data
    prices = {"AAPL": 178.50, "GOOGL": 140.25, "MSFT": 378.90}
    return {
        "symbol": symbol,
        "price": prices.get(symbol.upper(), 0),
        "currency": "USD"
    }

def calculate_mortgage(principal: float, rate: float, years: int) -> Dict[str, Any]:
    """Calculate monthly mortgage payment."""
    monthly_rate = rate / 100 / 12
    num_payments = years * 12
    
    if monthly_rate == 0:
        monthly_payment = principal / num_payments
    else:
        monthly_payment = principal * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    
    return {
        "monthly_payment": round(monthly_payment, 2),
        "total_payment": round(monthly_payment * num_payments, 2),
        "total_interest": round(monthly_payment * num_payments - principal, 2)
    }

# Create agent and register functions
agent = FunctionCallingAgent()

agent.register_function(
    func=get_stock_price,
    description="Get the current stock price for a given symbol",
    parameters={
        "type": "object",
        "properties": {
            "symbol": {
                "type": "string",
                "description": "Stock ticker symbol (e.g., AAPL, GOOGL)"
            }
        },
        "required": ["symbol"]
    }
)

agent.register_function(
    func=calculate_mortgage,
    description="Calculate monthly mortgage payment",
    parameters={
        "type": "object",
        "properties": {
            "principal": {
                "type": "number",
                "description": "Loan amount in dollars"
            },
            "rate": {
                "type": "number",
                "description": "Annual interest rate as a percentage"
            },
            "years": {
                "type": "integer",
                "description": "Loan term in years"
            }
        },
        "required": ["principal", "rate", "years"]
    }
)

# Test it
print(agent.process_query("What's Apple's stock price?"))
print(agent.process_query("Calculate mortgage for $400,000 at 7% for 30 years"))
```

## Function Calling with Anthropic Claude

```python
import anthropic
import json

client = anthropic.Anthropic()

# Define tools for Claude
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["location"]
        }
    }
]

def chat_with_tools(user_message: str) -> str:
    """Chat with Claude using tools."""
    
    messages = [{"role": "user", "content": user_message}]
    
    # Initial request
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        tools=tools,
        messages=messages
    )
    
    # Check if Claude wants to use a tool
    if response.stop_reason == "tool_use":
        # Find the tool use block
        tool_use = next(
            block for block in response.content 
            if block.type == "tool_use"
        )
        
        # Execute the tool
        if tool_use.name == "get_weather":
            result = get_weather(**tool_use.input)
        
        # Send result back to Claude
        messages.append({"role": "assistant", "content": response.content})
        messages.append({
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": json.dumps(result)
            }]
        })
        
        # Get final response
        final_response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=messages
        )
        
        return final_response.content[0].text
    
    return response.content[0].text
```

## Tool Choice Options

Control when the LLM uses functions:

```python
# Let LLM decide (default)
tool_choice = "auto"

# Force the LLM to use a specific function
tool_choice = {"type": "function", "function": {"name": "get_weather"}}

# Force the LLM to use some function (any of the available ones)
tool_choice = "required"

# Prevent function calling entirely
tool_choice = "none"

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice=tool_choice
)
```

## Parallel Function Calls

LLMs can request multiple function calls at once:

```python
# User: "What's the weather in Paris and London?"

# LLM response might include:
tool_calls = [
    {
        "id": "call_1",
        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}
    },
    {
        "id": "call_2", 
        "function": {"name": "get_weather", "arguments": '{"location": "London"}'}
    }
]

# Handle parallel calls
import asyncio

async def process_parallel_calls(tool_calls):
    """Process multiple tool calls in parallel."""
    
    async def execute_one(tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # Execute function (make it async if possible)
        result = await async_get_weather(**args)
        
        return {
            "tool_call_id": tool_call.id,
            "content": json.dumps(result)
        }
    
    # Execute all calls concurrently
    results = await asyncio.gather(*[
        execute_one(tc) for tc in tool_calls
    ])
    
    return results
```

## Best Practices

```yaml
function_definitions:
  - "Write clear, detailed descriptions"
  - "Use descriptive parameter names"
  - "Provide examples in descriptions"
  - "Mark required vs optional parameters"
  - "Use enums when values are constrained"

error_handling:
  - "Always validate function arguments"
  - "Return structured error messages"
  - "Handle timeouts gracefully"
  - "Log function calls for debugging"

performance:
  - "Execute parallel calls when possible"
  - "Cache function results when appropriate"
  - "Set reasonable timeouts"
  - "Limit number of tool calls per request"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│               Function Calling Basics - Summary                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  What it is:                                                             │
│    LLM requests to execute specific functions with arguments            │
│                                                                          │
│  The Flow:                                                               │
│    1. User query → LLM with tools defined                               │
│    2. LLM decides to call function(s)                                   │
│    3. Your code executes function(s)                                    │
│    4. Results sent back to LLM                                          │
│    5. LLM generates final response                                      │
│                                                                          │
│  Key Points:                                                             │
│    • YOU execute the functions, not the LLM                             │
│    • LLM only generates the function name and arguments                 │
│    • Multiple functions can be called in parallel                       │
│    • tool_choice controls when functions are used                       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Tool Schemas](/learn/agents/tool-use/tool-schemas) →
