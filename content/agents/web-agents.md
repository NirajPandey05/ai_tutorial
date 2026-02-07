# Web Browsing Agents

Learn how to build agents that can browse the web, extract information, and interact with websites.

## What are Web Agents?

Web agents can autonomously navigate websites, extract information, fill forms, and perform web-based tasks.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Web Agent Capabilities                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                        WEB AGENT                                 │   │
│   │                                                                  │   │
│   │  Can:                           Cannot (usually):               │   │
│   │  ✓ Navigate to URLs             ✗ Bypass CAPTCHAs              │   │
│   │  ✓ Extract page content         ✗ Access authenticated         │   │
│   │  ✓ Click links/buttons            content (without creds)      │   │
│   │  ✓ Fill out forms               ✗ Handle complex JS apps       │   │
│   │  ✓ Take screenshots               perfectly                    │   │
│   │  ✓ Search the web               ✗ Bypass rate limits           │   │
│   │  ✓ Parse structured data                                       │   │
│   │  ✓ Compare information                                         │   │
│   │                                                                  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Web Agent Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Web Agent Architecture                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│                         ┌───────────────┐                               │
│                         │   LLM Brain   │                               │
│                         │  (Reasoning)  │                               │
│                         └───────┬───────┘                               │
│                                 │                                        │
│            ┌────────────────────┼────────────────────┐                  │
│            ▼                    ▼                    ▼                  │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐        │
│   │  Web Search     │  │  Page Fetcher   │  │  Browser        │        │
│   │  (Search APIs)  │  │  (HTTP/HTML)    │  │  Automation     │        │
│   │                 │  │                 │  │  (Playwright)   │        │
│   │ • DuckDuckGo    │  │ • httpx/aiohttp │  │ • Click/Type    │        │
│   │ • Google API    │  │ • BeautifulSoup │  │ • Screenshot    │        │
│   │ • Bing API      │  │ • Trafilatura   │  │ • Navigate      │        │
│   └─────────────────┘  └─────────────────┘  └─────────────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Building Web Tools

### 1. Web Search Tool

```python
import httpx
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

class WebSearchTool:
    """Search the web using DuckDuckGo."""
    
    name = "web_search"
    description = """Search the web for information.
Use when you need current information, facts, or to find specific websites.
Returns titles, URLs, and snippets of top results."""
    
    async def run(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Execute a web search."""
        
        async with httpx.AsyncClient() as client:
            # DuckDuckGo HTML search
            response = await client.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10.0
            )
            
            results = self._parse_results(response.text, num_results)
            return results
    
    def _parse_results(self, html: str, limit: int) -> List[SearchResult]:
        """Parse search results from HTML."""
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, "html.parser")
        results = []
        
        for result in soup.select(".result")[:limit]:
            title_elem = result.select_one(".result__title")
            url_elem = result.select_one(".result__url")
            snippet_elem = result.select_one(".result__snippet")
            
            if title_elem and url_elem:
                results.append(SearchResult(
                    title=title_elem.get_text(strip=True),
                    url=url_elem.get_text(strip=True),
                    snippet=snippet_elem.get_text(strip=True) if snippet_elem else ""
                ))
        
        return results
```

### 2. Page Content Extractor

```python
import httpx
from bs4 import BeautifulSoup
from typing import Optional
import re

class PageContentTool:
    """Extract content from web pages."""
    
    name = "read_webpage"
    description = """Read and extract the main content from a webpage.
Use when you need to read the contents of a specific URL.
Returns the main text content, stripped of navigation and ads."""
    
    async def run(
        self, 
        url: str, 
        extract_links: bool = False,
        max_length: int = 5000
    ) -> dict:
        """Fetch and extract page content."""
        
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; WebAgent/1.0)"
                },
                timeout=15.0
            )
            
            response.raise_for_status()
            
            content = self._extract_content(response.text)
            
            result = {
                "url": str(response.url),
                "title": self._extract_title(response.text),
                "content": content[:max_length],
                "truncated": len(content) > max_length
            }
            
            if extract_links:
                result["links"] = self._extract_links(response.text, str(response.url))
            
            return result
    
    def _extract_title(self, html: str) -> str:
        """Extract page title."""
        soup = BeautifulSoup(html, "html.parser")
        title = soup.find("title")
        return title.get_text(strip=True) if title else ""
    
    def _extract_content(self, html: str) -> str:
        """Extract main content from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", 
                           "aside", "form", "iframe", "noscript"]):
            element.decompose()
        
        # Try to find main content
        main = (
            soup.find("main") or 
            soup.find("article") or 
            soup.find(id=re.compile(r"content|main|article", re.I)) or
            soup.find(class_=re.compile(r"content|main|article", re.I)) or
            soup.body
        )
        
        if main:
            text = main.get_text(separator="\n", strip=True)
            # Clean up whitespace
            text = re.sub(r'\n{3,}', '\n\n', text)
            return text
        
        return soup.get_text(separator="\n", strip=True)
    
    def _extract_links(self, html: str, base_url: str) -> List[dict]:
        """Extract links from page."""
        from urllib.parse import urljoin
        
        soup = BeautifulSoup(html, "html.parser")
        links = []
        
        for a in soup.find_all("a", href=True)[:20]:  # Limit to 20 links
            href = a["href"]
            if href.startswith(("#", "javascript:", "mailto:")):
                continue
            
            full_url = urljoin(base_url, href)
            text = a.get_text(strip=True)
            
            if text and len(text) > 2:
                links.append({
                    "text": text[:100],
                    "url": full_url
                })
        
        return links
```

### 3. Browser Automation Tool (Playwright)

```python
from playwright.async_api import async_playwright, Page, Browser
from typing import Optional, List
import asyncio

class BrowserTool:
    """Browser automation using Playwright."""
    
    name = "browser"
    description = """Control a web browser to interact with pages.
Use for: clicking buttons, filling forms, taking screenshots,
navigating JavaScript-heavy sites."""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None
    
    async def start(self, headless: bool = True):
        """Start the browser."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=headless)
        self.page = await self.browser.new_page()
    
    async def stop(self):
        """Stop the browser."""
        if self.browser:
            await self.browser.close()
    
    async def navigate(self, url: str) -> dict:
        """Navigate to a URL."""
        await self.page.goto(url, wait_until="networkidle")
        
        return {
            "url": self.page.url,
            "title": await self.page.title()
        }
    
    async def click(self, selector: str) -> dict:
        """Click an element."""
        try:
            await self.page.click(selector, timeout=5000)
            await self.page.wait_for_load_state("networkidle")
            return {"success": True, "message": f"Clicked {selector}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def fill(self, selector: str, text: str) -> dict:
        """Fill a text input."""
        try:
            await self.page.fill(selector, text)
            return {"success": True, "message": f"Filled {selector} with text"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def screenshot(self, path: str = None) -> bytes:
        """Take a screenshot."""
        return await self.page.screenshot(path=path, full_page=True)
    
    async def get_text(self, selector: str = "body") -> str:
        """Get text content of an element."""
        element = await self.page.query_selector(selector)
        if element:
            return await element.text_content()
        return ""
    
    async def get_elements(self, selector: str) -> List[dict]:
        """Get information about elements matching selector."""
        elements = await self.page.query_selector_all(selector)
        
        results = []
        for elem in elements[:20]:  # Limit results
            results.append({
                "text": await elem.text_content(),
                "tag": await elem.evaluate("el => el.tagName"),
                "href": await elem.get_attribute("href"),
            })
        
        return results
    
    async def execute_script(self, script: str) -> any:
        """Execute JavaScript on the page."""
        return await self.page.evaluate(script)
```

## Complete Web Agent Implementation

```python
from typing import Dict, Any, List
from openai import AsyncOpenAI
import json

class WebAgent:
    """Agent specialized for web browsing tasks."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        
        # Initialize tools
        self.search = WebSearchTool()
        self.reader = PageContentTool()
        self.browser = BrowserTool()
        
        self.system_prompt = """You are a web browsing agent that can search the web and read pages.

Available Tools:
- web_search(query, num_results): Search the web
- read_webpage(url, extract_links): Read a webpage's content
- browser_navigate(url): Navigate browser to URL
- browser_click(selector): Click an element
- browser_fill(selector, text): Fill a form field
- browser_screenshot(): Take a screenshot

Strategy:
1. Start with a search to find relevant URLs
2. Read promising pages to gather information
3. Use browser automation for interactive tasks
4. Synthesize information into a clear answer

Always cite your sources with URLs."""
    
    async def run(self, task: str) -> str:
        """Execute a web browsing task."""
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task}
        ]
        
        tools = self._get_tool_definitions()
        
        for _ in range(10):  # Max iterations
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            
            message = response.choices[0].message
            
            if not message.tool_calls:
                return message.content
            
            messages.append(message)
            
            for tool_call in message.tool_calls:
                result = await self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, default=str)
                })
        
        return "Max iterations reached"
    
    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name."""
        
        if name == "web_search":
            results = await self.search.run(**args)
            return [{"title": r.title, "url": r.url, "snippet": r.snippet} for r in results]
        
        elif name == "read_webpage":
            return await self.reader.run(**args)
        
        elif name == "browser_navigate":
            return await self.browser.navigate(args["url"])
        
        elif name == "browser_click":
            return await self.browser.click(args["selector"])
        
        elif name == "browser_fill":
            return await self.browser.fill(args["selector"], args["text"])
        
        elif name == "browser_screenshot":
            await self.browser.screenshot(path="screenshot.png")
            return {"saved": "screenshot.png"}
        
        return {"error": f"Unknown tool: {name}"}
    
    def _get_tool_definitions(self) -> List[dict]:
        """Get OpenAI tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "num_results": {"type": "integer", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_webpage",
                    "description": "Read content from a webpage",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to read"},
                            "extract_links": {"type": "boolean", "default": False}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_navigate",
                    "description": "Navigate browser to a URL",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_click",
                    "description": "Click an element on the page",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string", "description": "CSS selector"}
                        },
                        "required": ["selector"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "browser_fill",
                    "description": "Fill a form field",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selector": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["selector", "text"]
                    }
                }
            }
        ]
```

## Web Agent Use Cases

### 1. Research Agent

```python
async def research_topic(agent: WebAgent, topic: str) -> str:
    """Research a topic using the web agent."""
    
    task = f"""Research the topic: "{topic}"

Your task:
1. Search for authoritative sources on this topic
2. Read at least 3 different sources
3. Synthesize the information into a comprehensive summary
4. Include citations for all facts

Format your response as:
## Summary
[Main findings]

## Key Points
- [Point 1] (Source: [URL])
- [Point 2] (Source: [URL])
...

## Sources
1. [Title] - [URL]
2. [Title] - [URL]
..."""
    
    return await agent.run(task)
```

### 2. Price Comparison Agent

```python
async def compare_prices(agent: WebAgent, product: str) -> str:
    """Compare prices for a product across websites."""
    
    task = f"""Find and compare prices for: "{product}"

Your task:
1. Search for this product on multiple shopping sites
2. Extract prices from at least 3 different sources
3. Note any differences in specifications or sellers
4. Recommend the best option based on price and reliability

Format:
| Store | Price | Notes |
|-------|-------|-------|
| ... | ... | ... |

Best Option: [Your recommendation]"""
    
    return await agent.run(task)
```

### 3. Form Automation Agent

```python
async def fill_application(agent: WebAgent, url: str, data: dict) -> str:
    """Fill out a web form with provided data."""
    
    task = f"""Fill out the form at {url}

Data to enter:
{json.dumps(data, indent=2)}

Steps:
1. Navigate to the form page
2. Identify all form fields
3. Fill each field with the appropriate data
4. Do NOT submit the form
5. Take a screenshot for verification
6. Report what was filled"""
    
    return await agent.run(task)
```

## Best Practices

```yaml
rate_limiting:
  - "Add delays between requests (1-2 seconds)"
  - "Respect robots.txt"
  - "Use rotating user agents"
  - "Handle 429 (rate limit) errors gracefully"

content_extraction:
  - "Handle different page structures"
  - "Fall back to full page text if main content fails"
  - "Limit content length to avoid token limits"
  - "Clean HTML entities and special characters"

error_handling:
  - "Handle network timeouts"
  - "Deal with CAPTCHA pages gracefully"
  - "Retry failed requests with backoff"
  - "Validate URLs before fetching"

security:
  - "Never store or log credentials"
  - "Sanitize extracted content"
  - "Be cautious with JavaScript execution"
  - "Validate file downloads"
```

## Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Web Browsing Agents - Summary                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Core Tools:                                                             │
│    • Web Search: Find relevant URLs                                     │
│    • Page Reader: Extract content from URLs                             │
│    • Browser Automation: Interactive tasks                              │
│                                                                          │
│  Use Cases:                                                              │
│    • Research and information gathering                                 │
│    • Price comparison                                                   │
│    • Form filling                                                       │
│    • Data extraction                                                    │
│                                                                          │
│  Key Libraries:                                                          │
│    • httpx/aiohttp: HTTP requests                                       │
│    • BeautifulSoup: HTML parsing                                        │
│    • Playwright: Browser automation                                     │
│                                                                          │
│  Important Considerations:                                               │
│    • Respect rate limits and robots.txt                                 │
│    • Handle errors gracefully                                           │
│    • Cite sources in responses                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

Next: [Code Generation Agents](/learn/agents/agent-capabilities/code-agents) →
