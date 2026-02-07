/**
 * Pyodide Sandbox - In-Browser Python Execution
 * 
 * This module provides a complete in-browser Python execution environment
 * using Pyodide (Python compiled to WebAssembly).
 */

class PyodideSandbox {
    constructor(options = {}) {
        this.pyodide = null;
        this.isReady = false;
        this.isLoading = false;
        this.loadingProgress = 0;
        this.onProgress = options.onProgress || (() => {});
        this.onReady = options.onReady || (() => {});
        this.onOutput = options.onOutput || (() => {});
        this.onError = options.onError || (() => {});
        this.timeout = options.timeout || 30000; // 30 second default timeout
        this.preloadPackages = options.preloadPackages || [];
        this.loadedPackages = new Set();
        
        // Output capture
        this.outputBuffer = [];
        this.errorBuffer = [];
    }

    /**
     * Initialize Pyodide and load the Python environment
     */
    async initialize() {
        if (this.isReady) return true;
        if (this.isLoading) {
            // Wait for existing load to complete
            return new Promise((resolve) => {
                const checkReady = setInterval(() => {
                    if (this.isReady) {
                        clearInterval(checkReady);
                        resolve(true);
                    }
                }, 100);
            });
        }

        this.isLoading = true;
        this.onProgress({ stage: 'loading', progress: 0, message: 'Loading Python runtime...' });

        try {
            // Load Pyodide
            this.pyodide = await loadPyodide({
                indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
                stdout: (text) => this.handleStdout(text),
                stderr: (text) => this.handleStderr(text),
            });

            this.onProgress({ stage: 'loading', progress: 50, message: 'Python runtime loaded' });

            // Set up Python environment
            await this.setupEnvironment();

            this.onProgress({ stage: 'loading', progress: 70, message: 'Setting up environment...' });

            // Preload requested packages
            if (this.preloadPackages.length > 0) {
                await this.loadPackages(this.preloadPackages);
            }

            this.isReady = true;
            this.isLoading = false;
            this.onProgress({ stage: 'ready', progress: 100, message: 'Ready' });
            this.onReady();

            return true;
        } catch (error) {
            this.isLoading = false;
            this.onError({ type: 'initialization', message: error.message });
            throw error;
        }
    }

    /**
     * Set up the Python environment with helper functions
     */
    async setupEnvironment() {
        // Create helper functions and mock modules
        await this.pyodide.runPythonAsync(`
import sys
import io
from contextlib import redirect_stdout, redirect_stderr

# Create a mock for common packages that need API keys
class MockOpenAI:
    """Mock OpenAI client for educational demos"""
    
    class ChatCompletion:
        @staticmethod
        def create(**kwargs):
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "[Mock Response] This is a simulated response. In a real environment with API keys, you would see actual LLM output here."
                    }
                }]
            }
    
    class Completion:
        @staticmethod
        def create(**kwargs):
            return {
                "choices": [{
                    "text": "[Mock Response] Simulated completion output."
                }]
            }

class MockAnthropic:
    """Mock Anthropic client for educational demos"""
    
    class Messages:
        @staticmethod
        def create(**kwargs):
            return type('Response', (), {
                'content': [type('Block', (), {'text': '[Mock Response] Simulated Claude response.'})()]
            })()

# Helper function to print formatted output
def print_section(title, content=""):
    """Print a formatted section header"""
    print(f"\\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")
    if content:
        print(content)

def print_success(message):
    """Print a success message"""
    print(f"✅ {message}")

def print_error(message):
    """Print an error message"""
    print(f"❌ {message}")

def print_info(message):
    """Print an info message"""
    print(f"ℹ️  {message}")

def print_warning(message):
    """Print a warning message"""
    print(f"⚠️  {message}")

# Utility function to simulate API delay
import asyncio

async def simulate_api_call(delay=0.5):
    """Simulate an API call with a delay"""
    await asyncio.sleep(delay)
    return True

print_info("Python sandbox environment initialized!")
print_info("Available helpers: print_section(), print_success(), print_error(), print_info(), print_warning()")
`);
    }

    /**
     * Handle stdout output
     */
    handleStdout(text) {
        this.outputBuffer.push(text);
        this.onOutput({ type: 'stdout', text });
    }

    /**
     * Handle stderr output
     */
    handleStderr(text) {
        this.errorBuffer.push(text);
        this.onOutput({ type: 'stderr', text });
    }

    /**
     * Load Python packages
     */
    async loadPackages(packages) {
        if (!this.isReady && !this.isLoading) {
            await this.initialize();
        }

        const packagesToLoad = packages.filter(pkg => !this.loadedPackages.has(pkg));
        
        if (packagesToLoad.length === 0) return;

        this.onProgress({ 
            stage: 'packages', 
            progress: 0, 
            message: `Loading packages: ${packagesToLoad.join(', ')}` 
        });

        try {
            await this.pyodide.loadPackagesFromImports(packagesToLoad.join('\n'));
            
            // Also try micropip for packages not in Pyodide
            await this.pyodide.runPythonAsync(`
import micropip
packages_to_install = ${JSON.stringify(packagesToLoad)}
for pkg in packages_to_install:
    try:
        __import__(pkg)
    except ImportError:
        try:
            await micropip.install(pkg)
        except Exception as e:
            print(f"Note: Could not install {pkg}: {e}")
`);

            packagesToLoad.forEach(pkg => this.loadedPackages.add(pkg));
            
            this.onProgress({ 
                stage: 'packages', 
                progress: 100, 
                message: 'Packages loaded' 
            });
        } catch (error) {
            console.warn('Package loading warning:', error);
            // Continue anyway - some packages might not be available
        }
    }

    /**
     * Execute Python code with timeout
     */
    async execute(code, options = {}) {
        if (!this.isReady) {
            await this.initialize();
        }

        // Clear output buffers
        this.outputBuffer = [];
        this.errorBuffer = [];

        const timeout = options.timeout || this.timeout;
        
        // Create a promise that rejects after timeout
        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Execution timed out')), timeout);
        });

        // Create the execution promise
        const executionPromise = this.executeCode(code);

        try {
            const result = await Promise.race([executionPromise, timeoutPromise]);
            return {
                success: true,
                output: this.outputBuffer.join(''),
                errors: this.errorBuffer.join(''),
                result: result
            };
        } catch (error) {
            return {
                success: false,
                output: this.outputBuffer.join(''),
                errors: this.errorBuffer.join('') || error.message,
                error: error.message
            };
        }
    }

    /**
     * Internal code execution
     */
    async executeCode(code) {
        try {
            // Check if code contains async/await
            const hasAsync = /\bawait\b/.test(code) || /\basync\s+def\b/.test(code);
            
            if (hasAsync) {
                // Wrap in async function if needed
                const wrappedCode = `
import asyncio

async def __main__():
${code.split('\n').map(line => '    ' + line).join('\n')}

asyncio.get_event_loop().run_until_complete(__main__())
`;
                return await this.pyodide.runPythonAsync(wrappedCode);
            } else {
                return await this.pyodide.runPythonAsync(code);
            }
        } catch (error) {
            // Format Python traceback nicely
            throw new Error(this.formatPythonError(error));
        }
    }

    /**
     * Format Python errors for display
     */
    formatPythonError(error) {
        let message = error.message || String(error);
        
        // Extract the most relevant part of the traceback
        const lines = message.split('\n');
        const relevantLines = [];
        let inUserCode = false;
        
        for (const line of lines) {
            if (line.includes('<exec>') || line.includes('File "<exec>"')) {
                inUserCode = true;
            }
            if (inUserCode || line.startsWith('  ') || /Error:|Exception:/.test(line)) {
                relevantLines.push(line);
            }
        }
        
        return relevantLines.length > 0 ? relevantLines.join('\n') : message;
    }

    /**
     * Get Python variable value
     */
    getVariable(name) {
        if (!this.isReady) return undefined;
        try {
            return this.pyodide.globals.get(name);
        } catch {
            return undefined;
        }
    }

    /**
     * Set Python variable
     */
    setVariable(name, value) {
        if (!this.isReady) return false;
        try {
            this.pyodide.globals.set(name, value);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Reset the Python environment
     */
    async reset() {
        if (!this.isReady) return;
        
        this.outputBuffer = [];
        this.errorBuffer = [];
        
        // Reset globals but keep helper functions
        await this.pyodide.runPythonAsync(`
# Clear user variables
import sys
_keep = {'__name__', '__doc__', '__package__', '__loader__', '__spec__', 
         '__builtins__', 'sys', 'io', 'print_section', 'print_success', 
         'print_error', 'print_info', 'print_warning', 'MockOpenAI', 
         'MockAnthropic', 'simulate_api_call', 'redirect_stdout', 'redirect_stderr'}
_to_delete = [name for name in dir() if not name.startswith('_') and name not in _keep]
for name in _to_delete:
    try:
        del globals()[name]
    except:
        pass
`);
    }

    /**
     * Check if a package is available
     */
    async isPackageAvailable(packageName) {
        if (!this.isReady) return false;
        try {
            await this.pyodide.runPythonAsync(`import ${packageName}`);
            return true;
        } catch {
            return false;
        }
    }

    /**
     * Get list of available packages
     */
    getLoadedPackages() {
        return Array.from(this.loadedPackages);
    }
}

/**
 * Sandbox Presets for different lab types
 */
const SandboxPresets = {
    // Basic Python - no special packages
    basic: {
        packages: [],
        setupCode: `
print_info("Basic Python environment ready!")
print_info("You can use standard Python libraries.")
`
    },

    // Data Science preset
    dataScience: {
        packages: ['numpy', 'pandas', 'matplotlib'],
        setupCode: `
import numpy as np
import pandas as pd
print_success("Data science packages loaded!")
print_info("Available: numpy (np), pandas (pd)")
`
    },

    // LLM/AI concepts (with mocks)
    llmBasics: {
        packages: ['json'],
        setupCode: `
import json

# Mock LLM client for educational purposes
openai = MockOpenAI()
anthropic = MockAnthropic()

print_success("LLM environment ready (using educational mocks)")
print_info("Available: openai, anthropic (mocked for demos)")
print_warning("For real API calls, use the server-side execution mode")
`
    },

    // RAG concepts
    rag: {
        packages: ['numpy'],
        setupCode: `
import numpy as np
import json

# Simple vector operations for RAG demos
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def euclidean_distance(a, b):
    """Calculate Euclidean distance between two vectors"""
    return np.linalg.norm(np.array(a) - np.array(b))

# Mock embedding function
def get_embedding(text, dim=384):
    """Generate a mock embedding for educational purposes"""
    # Use hash of text to generate consistent pseudo-random embedding
    np.random.seed(hash(text) % (2**32))
    embedding = np.random.randn(dim)
    return embedding / np.linalg.norm(embedding)  # Normalize

print_success("RAG environment ready!")
print_info("Available: cosine_similarity(), euclidean_distance(), get_embedding()")
`
    },

    // Tokenization concepts
    tokenization: {
        packages: [],
        setupCode: `
# Simple tokenization demonstrations
def simple_tokenize(text):
    """Basic whitespace tokenization"""
    return text.split()

def char_tokenize(text):
    """Character-level tokenization"""
    return list(text)

def bpe_demo(text, vocab_size=100):
    """Demonstrate BPE-like tokenization concept"""
    # This is a simplified demonstration
    tokens = list(text)
    print_info(f"Starting with {len(tokens)} character tokens")
    
    # Show the concept of merging frequent pairs
    from collections import Counter
    pairs = Counter(zip(tokens[:-1], tokens[1:]))
    most_common = pairs.most_common(3)
    
    print_info("Most common adjacent pairs:")
    for pair, count in most_common:
        print(f"  '{pair[0]}' + '{pair[1]}' appears {count} times")
    
    return tokens

print_success("Tokenization environment ready!")
print_info("Available: simple_tokenize(), char_tokenize(), bpe_demo()")
`
    },

    // Prompt engineering
    promptEngineering: {
        packages: ['json'],
        setupCode: `
import json

# Mock LLM for prompt engineering practice
class PromptTester:
    """Test prompts with simulated responses"""
    
    def __init__(self):
        self.history = []
    
    def test(self, prompt, expected_behavior=""):
        """Test a prompt and analyze its structure"""
        self.history.append(prompt)
        
        analysis = {
            "length": len(prompt),
            "word_count": len(prompt.split()),
            "has_examples": "example" in prompt.lower() or "e.g." in prompt.lower(),
            "has_constraints": any(word in prompt.lower() for word in ["must", "should", "only", "don't", "never"]),
            "has_role": any(word in prompt.lower() for word in ["you are", "act as", "behave as"]),
            "has_format": any(word in prompt.lower() for word in ["json", "list", "format", "structure"]),
        }
        
        print_section("Prompt Analysis")
        print(f"Length: {analysis['length']} characters, {analysis['word_count']} words")
        print(f"Has examples: {'✓' if analysis['has_examples'] else '✗'}")
        print(f"Has constraints: {'✓' if analysis['has_constraints'] else '✗'}")
        print(f"Has role definition: {'✓' if analysis['has_role'] else '✗'}")
        print(f"Has format specification: {'✓' if analysis['has_format'] else '✗'}")
        
        return analysis

prompt_tester = PromptTester()
print_success("Prompt engineering environment ready!")
print_info("Available: prompt_tester.test(prompt)")
`
    },

    // Agent concepts
    agents: {
        packages: ['json'],
        setupCode: `
import json
from typing import Callable, Dict, Any, List

# Simple tool registry for agent demos
class ToolRegistry:
    """Register and manage agent tools"""
    
    def __init__(self):
        self.tools: Dict[str, Dict] = {}
    
    def register(self, name: str, description: str, parameters: Dict = None):
        """Decorator to register a tool"""
        def decorator(func: Callable):
            self.tools[name] = {
                "name": name,
                "description": description,
                "parameters": parameters or {},
                "function": func
            }
            return func
        return decorator
    
    def get_tool_schema(self) -> List[Dict]:
        """Get OpenAI-style tool schemas"""
        schemas = []
        for name, tool in self.tools.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            })
        return schemas
    
    def execute(self, name: str, **kwargs):
        """Execute a tool by name"""
        if name not in self.tools:
            return {"error": f"Tool '{name}' not found"}
        return self.tools[name]["function"](**kwargs)

# Create global tool registry
tools = ToolRegistry()

# Example tools
@tools.register("calculator", "Perform basic math operations", {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "Math expression to evaluate"}
    }
})
def calculator(expression: str):
    try:
        # Safe eval for basic math
        allowed = set('0123456789+-*/.() ')
        if all(c in allowed for c in expression):
            return {"result": eval(expression)}
        return {"error": "Invalid expression"}
    except Exception as e:
        return {"error": str(e)}

# Simple agent loop demonstration
class SimpleAgent:
    """Demonstrate basic agent loop concepts"""
    
    def __init__(self, tools: ToolRegistry):
        self.tools = tools
        self.messages = []
        self.max_iterations = 5
    
    def think(self, observation: str) -> str:
        """Agent reasoning step (simplified)"""
        print_info(f"Thinking about: {observation[:50]}...")
        return "I should use a tool to help with this."
    
    def act(self, thought: str) -> Dict:
        """Agent action step (simplified)"""
        print_info(f"Taking action based on thought...")
        return {"tool": "calculator", "args": {"expression": "2+2"}}
    
    def observe(self, action_result: Any) -> str:
        """Process action result"""
        return f"Tool returned: {action_result}"
    
    def run(self, task: str):
        """Run the agent loop"""
        print_section("Agent Loop Demo")
        print(f"Task: {task}")
        
        for i in range(self.max_iterations):
            print(f"\\n--- Iteration {i+1} ---")
            thought = self.think(task)
            print(f"Thought: {thought}")
            
            action = self.act(thought)
            print(f"Action: {action}")
            
            result = self.tools.execute(action["tool"], **action["args"])
            print(f"Result: {result}")
            
            observation = self.observe(result)
            print(f"Observation: {observation}")
            
            # Demo exits after one loop
            print_success("Agent loop iteration complete!")
            break

agent = SimpleAgent(tools)
print_success("Agent environment ready!")
print_info("Available: tools (ToolRegistry), agent (SimpleAgent)")
print_info("Try: agent.run('Calculate 2+2')")
`
    }
};

/**
 * Create a sandbox with a preset configuration
 */
async function createSandbox(presetName = 'basic', options = {}) {
    const preset = SandboxPresets[presetName] || SandboxPresets.basic;
    
    const sandbox = new PyodideSandbox({
        ...options,
        preloadPackages: [...(preset.packages || []), ...(options.preloadPackages || [])]
    });
    
    await sandbox.initialize();
    
    // Run preset setup code
    if (preset.setupCode) {
        await sandbox.execute(preset.setupCode);
    }
    
    return sandbox;
}

// Export for use in other modules
window.PyodideSandbox = PyodideSandbox;
window.SandboxPresets = SandboxPresets;
window.createSandbox = createSandbox;
