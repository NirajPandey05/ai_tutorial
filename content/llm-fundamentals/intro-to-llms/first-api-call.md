# Lab: Your First LLM API Call

In this hands-on lab, you'll make your first API call to an LLM. You'll see how easy it is to integrate AI into your applications!

## Prerequisites

Before starting, make sure you have:
- ✅ At least one API key configured in [Settings](/settings)
- ✅ Basic Python knowledge

## What You'll Build

A simple Python script that:
1. Connects to an LLM provider
2. Sends a message
3. Receives and displays the response

## Understanding the API

All major LLM providers follow a similar pattern:

```python
# The basic pattern for all providers
response = client.chat(
    model="model-name",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Your message here"}
    ]
)
print(response.content)
```

### Message Roles

| Role | Purpose | Example |
|------|---------|---------|
| `system` | Sets the AI's behavior | "You are a helpful coding assistant" |
| `user` | Your messages to the AI | "How do I read a file in Python?" |
| `assistant` | The AI's responses | "You can use open()..." |

## The Lab Exercise

Your task: Create a simple "Ask the AI" program that:
1. Takes a question as input
2. Sends it to the LLM
3. Displays the response

### Starter Code

```python
# Your first LLM API call!
# Try modifying the question and see how the AI responds

question = "What is the capital of France?"

# The API call will be made when you click Run
response = ask_llm(question)
print(f"Question: {question}")
print(f"Answer: {response}")
```

### Expected Output

```
Question: What is the capital of France?
Answer: The capital of France is Paris. Paris is not only the capital 
but also the largest city in France, known for landmarks like the 
Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral.
```

## Experiments to Try

Once you get the basic call working, try these variations:

### 1. Change the Question
```python
question = "Explain quantum computing in simple terms"
```

### 2. Ask for Code
```python
question = "Write a Python function to calculate factorial"
```

### 3. Get Creative
```python
question = "Write a haiku about programming"
```

### 4. Ask for Structured Output
```python
question = "List the top 5 programming languages as JSON"
```

## Behind the Scenes

When you make an API call, here's what happens:

```
┌─────────────────┐     HTTPS Request      ┌─────────────────┐
│                 │  ─────────────────────> │                 │
│   Your Code     │                         │   LLM Server    │
│                 │  <───────────────────── │                 │
└─────────────────┘     JSON Response       └─────────────────┘

Request:
{
    "model": "gpt-5.2",
    "messages": [
        {"role": "user", "content": "What is AI?"}
    ]
}

Response:
{
    "choices": [{
        "message": {
            "role": "assistant",
            "content": "AI stands for Artificial Intelligence..."
        }
    }],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 45,
        "total_tokens": 57
    }
}
```

## Common Issues

### API Key Not Set
```
Error: API key not configured
```
**Solution**: Go to [Settings](/settings) and add your API key.

### Rate Limiting
```
Error: Rate limit exceeded
```
**Solution**: Wait a moment and try again. Consider upgrading your API plan for higher limits.

### Invalid Request
```
Error: Invalid request format
```
**Solution**: Check that your message is properly formatted.

## What's Next?

Now that you've made your first API call, you're ready to:

1. **Prompt Engineering** - Learn techniques to get better responses
2. **Parameters** - Control temperature, max tokens, and more
3. **Streaming** - Get responses as they're generated

## Success Criteria

✅ Successfully sent a message to an LLM  
✅ Received and displayed a response  
✅ Tried at least 3 different questions  
✅ Understood the basic API structure  

---

**Congratulations!** 🎉 You've made your first LLM API call. This is the foundation for everything else you'll build in AI engineering!
