# Function Calling Best Practices

Learn when and how to use function calling effectively, including design patterns, error handling, and optimization strategies.

---

## Function Calling vs. Structured Outputs

Understanding when to use each approach:

| Feature | Function Calling | Structured Outputs |
|---------|-----------------|-------------------|
| **Purpose** | Execute actions, retrieve data | Parse/extract information |
| **Output** | Function name + arguments | Data conforming to schema |
| **Control Flow** | Model decides which function | Always returns data |
| **Use Case** | Tools, APIs, actions | Extraction, classification |

```python
# Function Calling: Model decides to call a function
# "What's the weather in Paris?" → calls get_weather(city="Paris")

# Structured Outputs: Model always returns structured data
# "Analyze this sentiment" → {"sentiment": "positive", "confidence": 0.9}
```

---

## Defining Effective Functions

### Basic Function Definition

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location. Use this when the user asks about weather conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City and state/country, e.g., 'San Francisco, CA' or 'London, UK'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit preference"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather like in Tokyo?"}],
    tools=tools,
    tool_choice="auto"  # Let model decide
)
```

### Function Description Best Practices

```python
# ✅ Good: Clear, specific description
{
    "name": "search_products",
    "description": """Search the product catalog by various criteria.
    
    Use this function when:
    - User wants to find products
    - User asks about product availability
    - User needs product recommendations
    
    Do NOT use for:
    - Order status inquiries (use get_order_status)
    - Account information (use get_account_info)
    """,
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search terms or product name"
            },
            "category": {
                "type": "string",
                "enum": ["electronics", "clothing", "home", "sports"],
                "description": "Filter by product category"
            },
            "max_price": {
                "type": "number",
                "description": "Maximum price in USD"
            },
            "in_stock_only": {
                "type": "boolean",
                "description": "Only return products currently in stock",
                "default": True
            }
        },
        "required": ["query"]
    }
}

# ❌ Bad: Vague description
{
    "name": "search",
    "description": "Search for things",
    "parameters": {
        "type": "object",
        "properties": {
            "q": {"type": "string"}
        }
    }
}
```

---

## Tool Choice Strategies

### Auto vs. Required vs. None

```python
# Auto: Model decides whether to call a function
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# Required: Model MUST call a function (any function)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="required"
)

# Specific function: Model MUST call this specific function
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "get_weather"}}
)

# None: Model cannot call functions (even if provided)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
    tool_choice="none"
)
```

### Dynamic Tool Selection

```python
def get_available_tools(user_context: dict) -> list:
    """Return tools based on user permissions and context"""
    
    base_tools = [search_products_tool, get_product_details_tool]
    
    if user_context.get("is_authenticated"):
        base_tools.extend([
            get_order_status_tool,
            get_account_info_tool
        ])
    
    if user_context.get("is_admin"):
        base_tools.extend([
            update_inventory_tool,
            manage_users_tool
        ])
    
    return base_tools

# Use dynamically based on user
tools = get_available_tools(current_user)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools
)
```

---

## Handling Function Calls

### Basic Execution Loop

```python
import json

def execute_function(name: str, arguments: dict) -> str:
    """Execute a function and return the result as a string"""
    
    functions = {
        "get_weather": get_weather,
        "search_products": search_products,
        "get_order_status": get_order_status
    }
    
    if name not in functions:
        return json.dumps({"error": f"Unknown function: {name}"})
    
    try:
        result = functions[name](**arguments)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def chat_with_functions(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        # Check if model wants to call functions
        if not assistant_message.tool_calls:
            # No function calls - return the response
            return assistant_message.content
        
        # Execute all function calls
        for tool_call in assistant_message.tool_calls:
            function_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            result = execute_function(function_name, arguments)
            
            # Add function result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
        
        # Continue loop - model will process function results
```

### Parallel Function Calls

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def execute_functions_parallel(tool_calls: list) -> list:
    """Execute multiple function calls in parallel"""
    
    async def execute_one(tool_call):
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Run blocking function in thread pool
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            result = await loop.run_in_executor(
                pool,
                lambda: execute_function(function_name, arguments)
            )
        
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "content": result
        }
    
    # Execute all in parallel
    results = await asyncio.gather(*[execute_one(tc) for tc in tool_calls])
    return results

# Usage in async context
async def chat_async(user_message: str) -> str:
    messages = [{"role": "user", "content": user_message}]
    
    while True:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools
        )
        
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        
        if not assistant_message.tool_calls:
            return assistant_message.content
        
        # Execute functions in parallel
        tool_results = await execute_functions_parallel(
            assistant_message.tool_calls
        )
        messages.extend(tool_results)
```

---

## Error Handling Patterns

### Graceful Error Responses

```python
def execute_function_safely(name: str, arguments: dict) -> str:
    """Execute function with comprehensive error handling"""
    
    try:
        # Validate function exists
        if name not in available_functions:
            return json.dumps({
                "success": False,
                "error": "function_not_found",
                "message": f"The function '{name}' is not available. Available functions: {list(available_functions.keys())}"
            })
        
        # Validate arguments
        func = available_functions[name]
        try:
            # Use function signature validation
            import inspect
            sig = inspect.signature(func)
            sig.bind(**arguments)
        except TypeError as e:
            return json.dumps({
                "success": False,
                "error": "invalid_arguments",
                "message": str(e)
            })
        
        # Execute function
        result = func(**arguments)
        
        return json.dumps({
            "success": True,
            "data": result
        })
        
    except TimeoutError:
        return json.dumps({
            "success": False,
            "error": "timeout",
            "message": "The operation timed out. Please try again."
        })
        
    except PermissionError:
        return json.dumps({
            "success": False,
            "error": "permission_denied",
            "message": "You don't have permission to perform this action."
        })
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": "internal_error",
            "message": f"An error occurred: {str(e)}"
        })
```

### Retry Logic

```python
import time
from functools import wraps

def with_retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator to add retry logic to functions"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts - 1:
                        time.sleep(delay * (attempt + 1))
            raise last_error
        return wrapper
    return decorator

@with_retry(max_attempts=3)
def get_weather(location: str, unit: str = "celsius") -> dict:
    """Get weather with automatic retry on failure"""
    response = weather_api.get(location, unit)
    return response.json()
```

---

## Function Design Patterns

### Composite Functions

Combine multiple operations into a single function:

```python
# Instead of multiple small functions...
tools_verbose = [
    {"name": "get_user_name", ...},
    {"name": "get_user_email", ...},
    {"name": "get_user_orders", ...}
]

# Create a composite function
{
    "name": "get_user_info",
    "description": "Get user information. Specify which fields to retrieve.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_id": {"type": "string"},
            "include": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["profile", "orders", "preferences", "payment_methods"]
                },
                "description": "Which information to include"
            }
        },
        "required": ["user_id", "include"]
    }
}

def get_user_info(user_id: str, include: list) -> dict:
    result = {"user_id": user_id}
    
    if "profile" in include:
        result["profile"] = get_user_profile(user_id)
    if "orders" in include:
        result["orders"] = get_user_orders(user_id, limit=5)
    if "preferences" in include:
        result["preferences"] = get_user_preferences(user_id)
    if "payment_methods" in include:
        result["payment_methods"] = get_payment_methods(user_id)
    
    return result
```

### Stateful Function Context

```python
class FunctionContext:
    """Maintain state across function calls"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_data = {}
        self.call_history = []
    
    def execute(self, name: str, arguments: dict) -> str:
        # Add context to all calls
        arguments["_user_id"] = self.user_id
        arguments["_session"] = self.session_data
        
        result = execute_function(name, arguments)
        
        # Track call history
        self.call_history.append({
            "function": name,
            "arguments": arguments,
            "result": result,
            "timestamp": time.time()
        })
        
        return result

# Usage
context = FunctionContext(user_id="user_123")

# All function calls have access to user context
result = context.execute("search_products", {"query": "laptop"})
```

### Function Chaining

```python
{
    "name": "book_trip",
    "description": """Book a complete trip with flights, hotel, and car.
    
    This function will:
    1. Search for flights
    2. Find hotels near destination
    3. Arrange car rental if needed
    4. Calculate total cost
    5. Create booking
    
    Use this for complete trip planning instead of individual booking functions.""",
    "parameters": {
        "type": "object",
        "properties": {
            "origin": {"type": "string"},
            "destination": {"type": "string"},
            "departure_date": {"type": "string", "format": "date"},
            "return_date": {"type": "string", "format": "date"},
            "travelers": {"type": "integer"},
            "include_car": {"type": "boolean", "default": False},
            "budget_max": {"type": "number"}
        },
        "required": ["origin", "destination", "departure_date", "return_date", "travelers"]
    }
}

async def book_trip(
    origin: str,
    destination: str,
    departure_date: str,
    return_date: str,
    travelers: int,
    include_car: bool = False,
    budget_max: float = None
) -> dict:
    """Chain multiple bookings into one operation"""
    
    # Step 1: Search flights
    flights = await search_flights(origin, destination, departure_date, return_date, travelers)
    best_flight = select_best_option(flights, budget_max)
    
    # Step 2: Find hotels
    hotels = await search_hotels(destination, departure_date, return_date, travelers)
    best_hotel = select_best_option(hotels, budget_max)
    
    # Step 3: Car rental (optional)
    car = None
    if include_car:
        cars = await search_cars(destination, departure_date, return_date)
        car = select_best_option(cars, budget_max)
    
    # Step 4: Calculate total
    total = best_flight["price"] + best_hotel["price"]
    if car:
        total += car["price"]
    
    # Step 5: Create booking
    if budget_max and total > budget_max:
        return {
            "success": False,
            "message": f"Trip cost ${total} exceeds budget ${budget_max}",
            "options": {"flight": best_flight, "hotel": best_hotel, "car": car}
        }
    
    booking = await create_booking(best_flight, best_hotel, car)
    
    return {
        "success": True,
        "booking_id": booking["id"],
        "total_cost": total,
        "details": {
            "flight": best_flight,
            "hotel": best_hotel,
            "car": car
        }
    }
```

---

## Anthropic Tool Use

Claude has similar but slightly different tool calling:

```python
from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a location",
        "input_schema": {  # Note: input_schema, not parameters
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country"
                }
            },
            "required": ["location"]
        }
    }
]

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}]
)

# Check for tool use
for content in response.content:
    if content.type == "tool_use":
        tool_name = content.name
        tool_input = content.input
        tool_use_id = content.id
        
        # Execute function
        result = execute_function(tool_name, tool_input)
        
        # Continue conversation with result
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            tools=tools,
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result
                        }
                    ]
                }
            ]
        )
```

---

## Best Practices Summary

### Do's ✅

1. **Write detailed descriptions** - Include when to use AND when not to use
2. **Use enums for constrained choices** - Reduces errors
3. **Provide default values** - Makes functions more flexible
4. **Handle errors gracefully** - Return informative error objects
5. **Use composite functions** - Reduce round trips
6. **Validate arguments** - Before execution

### Don'ts ❌

1. **Don't use vague names** - `do_thing` → `update_user_profile`
2. **Don't skip descriptions** - Models need guidance
3. **Don't return raw exceptions** - Wrap in structured errors
4. **Don't create too many similar functions** - Confuses the model
5. **Don't forget idempotency** - Functions may be called multiple times

---

## Next Steps

- [Pydantic Integration](/learn/advanced-topics/structured-outputs/pydantic-integration) - Type-safe function arguments
- [Structured Outputs Lab](/learn/advanced-topics/structured-outputs/type-safe-lab) - Build a complete tool system
