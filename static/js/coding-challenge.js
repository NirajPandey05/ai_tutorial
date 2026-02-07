/**
 * Coding Challenge Engine - Auto-graded coding exercises
 * Uses Pyodide for in-browser Python execution with test case validation
 */

class CodingChallenge {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            maxAttempts: 10,
            showHints: true,
            timeLimit: null, // in seconds, null = no limit
            ...options
        };
        
        this.challenge = null;
        this.code = '';
        this.attempts = 0;
        this.hintsUsed = 0;
        this.solved = false;
        this.startTime = null;
        this.pyodide = null;
        this.editor = null;
    }
    
    /**
     * Load a challenge
     */
    async loadChallenge(challenge) {
        this.challenge = challenge;
        this.code = challenge.starterCode || '';
        this.attempts = 0;
        this.hintsUsed = 0;
        this.solved = false;
        this.startTime = Date.now();
        
        // Initialize Pyodide
        await this.initPyodide();
        
        this.render();
    }
    
    /**
     * Initialize Pyodide for Python execution
     */
    async initPyodide() {
        if (this.pyodide) return;
        
        try {
            this.pyodide = await loadPyodide();
            console.log('Pyodide loaded for coding challenges');
        } catch (error) {
            console.error('Failed to load Pyodide:', error);
        }
    }
    
    /**
     * Render the challenge interface
     */
    render() {
        if (!this.container || !this.challenge) return;
        
        const c = this.challenge;
        
        this.container.innerHTML = `
            <div class="challenge-container">
                <!-- Challenge Header -->
                <div class="challenge-header bg-slate-800 rounded-xl p-6 mb-6">
                    <div class="flex items-start justify-between">
                        <div>
                            <div class="flex items-center gap-3 mb-2">
                                <span class="text-2xl">${c.icon || '💻'}</span>
                                <h2 class="text-xl font-bold text-white">${c.title}</h2>
                                <span class="px-2 py-1 rounded text-xs font-medium ${this.getDifficultyClass(c.difficulty)}">
                                    ${c.difficulty}
                                </span>
                            </div>
                            <p class="text-slate-300 mb-4">${c.description}</p>
                            
                            ${c.examples ? `
                            <div class="mt-4">
                                <h4 class="text-sm font-semibold text-slate-400 mb-2">Examples:</h4>
                                <div class="space-y-2">
                                    ${c.examples.map(ex => `
                                        <div class="bg-slate-900 rounded-lg p-3 font-mono text-sm">
                                            <div class="text-slate-500">Input: <span class="text-cyan-400">${this.formatValue(ex.input)}</span></div>
                                            <div class="text-slate-500">Output: <span class="text-green-400">${this.formatValue(ex.output)}</span></div>
                                            ${ex.explanation ? `<div class="text-slate-600 text-xs mt-1"># ${ex.explanation}</div>` : ''}
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                            ` : ''}
                        </div>
                        
                        <div class="text-right">
                            <div class="text-sm text-slate-400">
                                <div>Attempts: <span class="text-white">${this.attempts}/${this.options.maxAttempts}</span></div>
                                ${this.options.showHints ? `<div>Hints: <span class="text-white">${this.hintsUsed}/${c.hints?.length || 0}</span></div>` : ''}
                            </div>
                            ${this.solved ? `
                                <div class="mt-2 px-3 py-1 bg-green-500/20 text-green-400 rounded-full text-sm font-medium">
                                    ✓ Solved
                                </div>
                            ` : ''}
                        </div>
                    </div>
                </div>
                
                <!-- Code Editor -->
                <div class="challenge-editor bg-slate-800 rounded-xl overflow-hidden mb-6">
                    <div class="flex items-center justify-between px-4 py-2 bg-slate-900 border-b border-slate-700">
                        <span class="text-sm text-slate-400">Solution</span>
                        <div class="flex items-center gap-2">
                            <button onclick="challenge.resetCode()" class="text-xs px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-slate-300">
                                Reset
                            </button>
                            ${this.options.showHints && c.hints && c.hints.length > this.hintsUsed ? `
                                <button onclick="challenge.showHint()" class="text-xs px-2 py-1 bg-yellow-500/20 hover:bg-yellow-500/30 rounded text-yellow-400">
                                    💡 Hint
                                </button>
                            ` : ''}
                        </div>
                    </div>
                    <div id="code-editor" class="h-64"></div>
                </div>
                
                <!-- Hints Display -->
                <div id="hints-container" class="mb-6 ${this.hintsUsed === 0 ? 'hidden' : ''}">
                    ${this.renderHints()}
                </div>
                
                <!-- Run Button -->
                <div class="flex items-center gap-4 mb-6">
                    <button 
                        onclick="challenge.runTests()"
                        class="px-6 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-medium transition-colors flex items-center gap-2"
                        ${this.solved || this.attempts >= this.options.maxAttempts ? 'disabled' : ''}>
                        <span>▶</span>
                        <span>Run Tests</span>
                    </button>
                    
                    <div id="run-status" class="text-sm text-slate-400"></div>
                </div>
                
                <!-- Test Results -->
                <div id="test-results" class="bg-slate-800 rounded-xl p-6">
                    <h3 class="text-lg font-semibold text-white mb-4">Test Cases</h3>
                    <div id="test-cases-list" class="space-y-3">
                        ${this.renderTestCases()}
                    </div>
                </div>
                
                ${this.solved ? this.renderSuccessMessage() : ''}
            </div>
        `;
        
        // Initialize code editor
        this.initEditor();
    }
    
    /**
     * Initialize Monaco editor or fallback
     */
    initEditor() {
        const editorContainer = document.getElementById('code-editor');
        if (!editorContainer) return;
        
        // Try Monaco first
        if (window.monaco) {
            this.editor = monaco.editor.create(editorContainer, {
                value: this.code,
                language: 'python',
                theme: 'vs-dark',
                minimap: { enabled: false },
                fontSize: 14,
                lineNumbers: 'on',
                automaticLayout: true,
                scrollBeyondLastLine: false,
                padding: { top: 16 }
            });
            
            this.editor.onDidChangeModelContent(() => {
                this.code = this.editor.getValue();
            });
        } else {
            // Fallback to textarea
            editorContainer.innerHTML = `
                <textarea 
                    id="code-textarea"
                    class="w-full h-full p-4 bg-slate-900 text-slate-100 font-mono text-sm resize-none focus:outline-none"
                    spellcheck="false"
                >${this.code}</textarea>
            `;
            
            const textarea = document.getElementById('code-textarea');
            textarea.addEventListener('input', () => {
                this.code = textarea.value;
            });
        }
    }
    
    /**
     * Render test cases list
     */
    renderTestCases() {
        if (!this.challenge?.testCases) return '<p class="text-slate-500">No test cases defined</p>';
        
        return this.challenge.testCases.map((tc, i) => {
            const result = tc.result;
            let statusClass = 'bg-slate-700';
            let statusIcon = '○';
            
            if (result === 'pass') {
                statusClass = 'bg-green-500/20 border-green-500/50';
                statusIcon = '✓';
            } else if (result === 'fail') {
                statusClass = 'bg-red-500/20 border-red-500/50';
                statusIcon = '✗';
            }
            
            const isHidden = tc.hidden && !this.solved;
            
            return `
                <div class="test-case p-4 rounded-lg border ${statusClass}">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center gap-3">
                            <span class="${result === 'pass' ? 'text-green-400' : result === 'fail' ? 'text-red-400' : 'text-slate-500'}">${statusIcon}</span>
                            <span class="font-medium ${isHidden ? 'text-slate-500' : 'text-white'}">
                                ${isHidden ? 'Hidden Test Case' : `Test ${i + 1}`}
                            </span>
                        </div>
                        ${tc.points ? `<span class="text-sm text-slate-400">${tc.points} pts</span>` : ''}
                    </div>
                    
                    ${!isHidden ? `
                        <div class="mt-3 font-mono text-sm">
                            <div class="text-slate-500">Input: <span class="text-cyan-400">${this.formatValue(tc.input)}</span></div>
                            <div class="text-slate-500">Expected: <span class="text-green-400">${this.formatValue(tc.expected)}</span></div>
                            ${tc.actual !== undefined ? `
                                <div class="text-slate-500">Got: <span class="${tc.result === 'pass' ? 'text-green-400' : 'text-red-400'}">${this.formatValue(tc.actual)}</span></div>
                            ` : ''}
                        </div>
                    ` : ''}
                    
                    ${tc.error ? `
                        <div class="mt-2 p-2 bg-red-500/10 rounded text-red-400 text-sm font-mono">
                            ${tc.error}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
    }
    
    /**
     * Render hints
     */
    renderHints() {
        if (!this.challenge?.hints || this.hintsUsed === 0) return '';
        
        return this.challenge.hints.slice(0, this.hintsUsed).map((hint, i) => `
            <div class="hint-card bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4 mb-3">
                <div class="flex items-center gap-2 mb-2">
                    <span>💡</span>
                    <span class="font-semibold text-yellow-400">Hint ${i + 1}</span>
                </div>
                <p class="text-slate-300">${hint}</p>
            </div>
        `).join('');
    }
    
    /**
     * Render success message
     */
    renderSuccessMessage() {
        const timeSpent = Math.round((Date.now() - this.startTime) / 1000);
        const minutes = Math.floor(timeSpent / 60);
        const seconds = timeSpent % 60;
        
        return `
            <div class="success-message mt-6 bg-green-500/10 border border-green-500/30 rounded-xl p-6 text-center">
                <div class="text-5xl mb-4">🎉</div>
                <h3 class="text-2xl font-bold text-green-400 mb-2">Challenge Complete!</h3>
                <p class="text-slate-300 mb-4">Great job solving this challenge!</p>
                
                <div class="flex justify-center gap-8 text-slate-400">
                    <div>
                        <div class="text-2xl font-bold text-white">${this.attempts}</div>
                        <div class="text-sm">Attempts</div>
                    </div>
                    <div>
                        <div class="text-2xl font-bold text-white">${minutes}:${seconds.toString().padStart(2, '0')}</div>
                        <div class="text-sm">Time</div>
                    </div>
                    <div>
                        <div class="text-2xl font-bold text-white">${this.hintsUsed}</div>
                        <div class="text-sm">Hints Used</div>
                    </div>
                </div>
                
                ${this.challenge.solution ? `
                    <button onclick="challenge.showSolution()" class="mt-6 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg text-sm">
                        View Reference Solution
                    </button>
                ` : ''}
            </div>
        `;
    }
    
    /**
     * Run test cases
     */
    async runTests() {
        if (!this.pyodide) {
            this.showStatus('Loading Python environment...', 'loading');
            await this.initPyodide();
        }
        
        this.attempts++;
        this.showStatus('Running tests...', 'loading');
        
        const testCases = this.challenge.testCases;
        let allPassed = true;
        
        for (let i = 0; i < testCases.length; i++) {
            const tc = testCases[i];
            
            try {
                // Prepare test code
                const testCode = `
${this.code}

# Run test
_result = ${this.challenge.functionName}(${this.formatPythonArgs(tc.input)})
_result
`;
                
                const result = await this.pyodide.runPythonAsync(testCode);
                const actual = result?.toJs ? result.toJs() : result;
                
                tc.actual = actual;
                tc.result = this.compareResults(actual, tc.expected) ? 'pass' : 'fail';
                tc.error = null;
                
                if (tc.result === 'fail') allPassed = false;
                
            } catch (error) {
                tc.result = 'fail';
                tc.error = error.message;
                tc.actual = undefined;
                allPassed = false;
            }
        }
        
        // Update UI
        document.getElementById('test-cases-list').innerHTML = this.renderTestCases();
        
        if (allPassed) {
            this.solved = true;
            this.showStatus('All tests passed! 🎉', 'success');
            this.saveChallengeResult();
            this.render(); // Re-render to show success message
        } else {
            const passedCount = testCases.filter(tc => tc.result === 'pass').length;
            this.showStatus(`${passedCount}/${testCases.length} tests passed`, 'error');
        }
    }
    
    /**
     * Compare test results
     */
    compareResults(actual, expected) {
        // Handle arrays/lists
        if (Array.isArray(expected)) {
            if (!Array.isArray(actual)) return false;
            if (actual.length !== expected.length) return false;
            return actual.every((v, i) => this.compareResults(v, expected[i]));
        }
        
        // Handle objects/dicts
        if (typeof expected === 'object' && expected !== null) {
            if (typeof actual !== 'object' || actual === null) return false;
            const keys = Object.keys(expected);
            if (keys.length !== Object.keys(actual).length) return false;
            return keys.every(k => this.compareResults(actual[k], expected[k]));
        }
        
        // Handle floats with tolerance
        if (typeof expected === 'number' && typeof actual === 'number') {
            return Math.abs(actual - expected) < 0.0001;
        }
        
        return actual === expected;
    }
    
    /**
     * Show a hint
     */
    showHint() {
        if (!this.challenge?.hints || this.hintsUsed >= this.challenge.hints.length) return;
        
        this.hintsUsed++;
        const hintsContainer = document.getElementById('hints-container');
        hintsContainer.classList.remove('hidden');
        hintsContainer.innerHTML = this.renderHints();
    }
    
    /**
     * Reset code to starter
     */
    resetCode() {
        this.code = this.challenge.starterCode || '';
        if (this.editor) {
            this.editor.setValue(this.code);
        } else {
            const textarea = document.getElementById('code-textarea');
            if (textarea) textarea.value = this.code;
        }
    }
    
    /**
     * Show reference solution
     */
    showSolution() {
        if (!this.challenge.solution) return;
        
        // Create modal
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4';
        modal.innerHTML = `
            <div class="bg-slate-800 rounded-xl max-w-2xl w-full max-h-[80vh] overflow-hidden">
                <div class="flex items-center justify-between px-6 py-4 border-b border-slate-700">
                    <h3 class="text-lg font-semibold text-white">Reference Solution</h3>
                    <button onclick="this.closest('.fixed').remove()" class="text-slate-400 hover:text-white">✕</button>
                </div>
                <div class="p-6 overflow-y-auto">
                    <pre class="bg-slate-900 rounded-lg p-4 overflow-x-auto"><code class="language-python">${this.escapeHtml(this.challenge.solution)}</code></pre>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
        
        // Highlight code
        if (window.hljs) {
            modal.querySelectorAll('pre code').forEach(block => hljs.highlightElement(block));
        }
    }
    
    /**
     * Show status message
     */
    showStatus(message, type) {
        const status = document.getElementById('run-status');
        if (!status) return;
        
        const colors = {
            loading: 'text-slate-400',
            success: 'text-green-400',
            error: 'text-red-400'
        };
        
        status.className = `text-sm ${colors[type] || 'text-slate-400'}`;
        status.innerHTML = type === 'loading' ? `<span class="animate-pulse">${message}</span>` : message;
    }
    
    /**
     * Save challenge result to localStorage
     */
    saveChallengeResult() {
        const challengeId = this.challenge.id || 'unknown';
        const results = JSON.parse(localStorage.getItem('challengeResults') || '{}');
        
        const timeSpent = Math.round((Date.now() - this.startTime) / 1000);
        
        if (!results[challengeId] || this.attempts < results[challengeId].bestAttempts) {
            results[challengeId] = {
                solved: true,
                bestAttempts: this.attempts,
                bestTime: timeSpent,
                hintsUsed: this.hintsUsed,
                lastCompleted: new Date().toISOString()
            };
            localStorage.setItem('challengeResults', JSON.stringify(results));
        }
    }
    
    // Helper methods
    getDifficultyClass(difficulty) {
        const classes = {
            'Easy': 'bg-green-500/20 text-green-400',
            'Medium': 'bg-yellow-500/20 text-yellow-400',
            'Hard': 'bg-red-500/20 text-red-400'
        };
        return classes[difficulty] || 'bg-slate-500/20 text-slate-400';
    }
    
    formatValue(value) {
        if (value === null) return 'None';
        if (value === undefined) return 'undefined';
        if (typeof value === 'string') return `"${value}"`;
        if (Array.isArray(value)) return `[${value.map(v => this.formatValue(v)).join(', ')}]`;
        if (typeof value === 'object') return JSON.stringify(value);
        return String(value);
    }
    
    formatPythonArgs(input) {
        if (Array.isArray(input)) {
            // Multiple arguments
            return input.map(v => this.toPythonValue(v)).join(', ');
        }
        return this.toPythonValue(input);
    }
    
    toPythonValue(value) {
        if (value === null) return 'None';
        if (typeof value === 'string') return `"${value}"`;
        if (typeof value === 'boolean') return value ? 'True' : 'False';
        if (Array.isArray(value)) return `[${value.map(v => this.toPythonValue(v)).join(', ')}]`;
        if (typeof value === 'object') return JSON.stringify(value).replace(/"/g, "'");
        return String(value);
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Sample challenges
const ChallengeData = {
    twoSum: {
        id: 'two-sum',
        title: 'Two Sum',
        icon: '🔢',
        difficulty: 'Easy',
        description: 'Given an array of integers and a target sum, return the indices of two numbers that add up to the target.',
        functionName: 'two_sum',
        starterCode: `def two_sum(nums, target):
    """
    Find two numbers that add up to target.
    
    Args:
        nums: List of integers
        target: Target sum
    
    Returns:
        List of two indices
    """
    # Your code here
    pass`,
        examples: [
            { input: [[2, 7, 11, 15], 9], output: [0, 1], explanation: '2 + 7 = 9' },
            { input: [[3, 2, 4], 6], output: [1, 2], explanation: '2 + 4 = 6' }
        ],
        testCases: [
            { input: [[2, 7, 11, 15], 9], expected: [0, 1] },
            { input: [[3, 2, 4], 6], expected: [1, 2] },
            { input: [[3, 3], 6], expected: [0, 1] },
            { input: [[1, 2, 3, 4, 5], 9], expected: [3, 4], hidden: true }
        ],
        hints: [
            'Think about using a hash map to store numbers you\'ve seen.',
            'For each number, check if (target - number) exists in your hash map.',
            'Store the index along with the number in your hash map.'
        ],
        solution: `def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []`
    },
    
    reverseString: {
        id: 'reverse-string',
        title: 'Reverse String',
        icon: '🔄',
        difficulty: 'Easy',
        description: 'Write a function that reverses a string without using built-in reverse methods.',
        functionName: 'reverse_string',
        starterCode: `def reverse_string(s):
    """
    Reverse the input string.
    
    Args:
        s: Input string
    
    Returns:
        Reversed string
    """
    # Your code here
    pass`,
        examples: [
            { input: 'hello', output: 'olleh' },
            { input: 'AI', output: 'IA' }
        ],
        testCases: [
            { input: 'hello', expected: 'olleh' },
            { input: 'AI', expected: 'IA' },
            { input: '', expected: '' },
            { input: 'a', expected: 'a' },
            { input: 'racecar', expected: 'racecar', hidden: true }
        ],
        hints: [
            'You can iterate through the string from the end to the beginning.',
            'Consider using two pointers or building a new string.'
        ],
        solution: `def reverse_string(s):
    result = ""
    for char in s:
        result = char + result
    return result`
    },
    
    countTokens: {
        id: 'count-tokens',
        title: 'Simple Token Counter',
        icon: '🔤',
        difficulty: 'Easy',
        description: 'Implement a simple token counter that splits text on whitespace and punctuation. This is a simplified version of how LLM tokenizers work.',
        functionName: 'count_tokens',
        starterCode: `def count_tokens(text):
    """
    Count tokens in text (split on whitespace and punctuation).
    
    Args:
        text: Input text string
    
    Returns:
        Number of tokens
    """
    # Your code here
    pass`,
        examples: [
            { input: 'Hello world', output: 2 },
            { input: 'Hello, world!', output: 4, explanation: 'Punctuation counts as separate tokens' }
        ],
        testCases: [
            { input: 'Hello world', expected: 2 },
            { input: 'Hello, world!', expected: 4 },
            { input: '', expected: 0 },
            { input: 'AI is amazing.', expected: 4 },
            { input: "It's a test!", expected: 5, hidden: true }
        ],
        hints: [
            'Use regular expressions to split on word boundaries.',
            'Filter out empty strings from the result.'
        ],
        solution: `import re

def count_tokens(text):
    if not text:
        return 0
    # Split on whitespace and keep punctuation as separate tokens
    tokens = re.findall(r"\\w+|[^\\w\\s]", text)
    return len(tokens)`
    },
    
    cosineSimilarity: {
        id: 'cosine-similarity',
        title: 'Cosine Similarity',
        icon: '📐',
        difficulty: 'Medium',
        description: 'Implement cosine similarity calculation between two vectors. This is the core metric used for semantic search in RAG systems.',
        functionName: 'cosine_similarity',
        starterCode: `def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector (list of floats)
        vec2: Second vector (list of floats)
    
    Returns:
        Cosine similarity (float between -1 and 1)
    """
    # Your code here
    pass`,
        examples: [
            { input: [[1, 0], [1, 0]], output: 1.0, explanation: 'Identical vectors = 1.0' },
            { input: [[1, 0], [0, 1]], output: 0.0, explanation: 'Perpendicular vectors = 0.0' }
        ],
        testCases: [
            { input: [[1, 0], [1, 0]], expected: 1.0 },
            { input: [[1, 0], [0, 1]], expected: 0.0 },
            { input: [[1, 2, 3], [1, 2, 3]], expected: 1.0 },
            { input: [[1, 1], [-1, -1]], expected: -1.0 },
            { input: [[3, 4], [4, 3]], expected: 0.96, hidden: true }
        ],
        hints: [
            'Cosine similarity = (A · B) / (||A|| × ||B||)',
            'Dot product: sum of element-wise multiplication',
            'Magnitude: square root of sum of squares'
        ],
        solution: `import math

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    
    if mag1 == 0 or mag2 == 0:
        return 0.0
    
    return dot_product / (mag1 * mag2)`
    }
};

// Export
window.CodingChallenge = CodingChallenge;
window.ChallengeData = ChallengeData;
