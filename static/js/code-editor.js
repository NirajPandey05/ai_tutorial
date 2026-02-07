/**
 * Code Editor Component
 * 
 * A feature-rich code editor built on Monaco Editor (VS Code's editor)
 * with Python syntax highlighting, autocomplete, and more.
 */

class CodeEditor {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;
        
        this.options = {
            language: options.language || 'python',
            theme: options.theme || 'vs-dark',
            fontSize: options.fontSize || 14,
            minimap: options.minimap !== false,
            lineNumbers: options.lineNumbers !== false,
            wordWrap: options.wordWrap || 'on',
            automaticLayout: options.automaticLayout !== false,
            readOnly: options.readOnly || false,
            value: options.value || '',
            onChange: options.onChange || (() => {}),
            onSave: options.onSave || (() => {}),
            ...options
        };
        
        this.editor = null;
        this.isReady = false;
        this.monacoLoaded = false;
    }

    /**
     * Load Monaco Editor from CDN
     */
    async loadMonaco() {
        if (this.monacoLoaded || window.monaco) {
            this.monacoLoaded = true;
            return;
        }

        return new Promise((resolve, reject) => {
            // Load Monaco loader
            const loaderScript = document.createElement('script');
            loaderScript.src = 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs/loader.js';
            loaderScript.onload = () => {
                // Configure Monaco paths
                window.require.config({
                    paths: {
                        vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs'
                    }
                });

                // Load Monaco
                window.require(['vs/editor/editor.main'], () => {
                    this.monacoLoaded = true;
                    this.setupMonacoThemes();
                    resolve();
                });
            };
            loaderScript.onerror = reject;
            document.head.appendChild(loaderScript);
        });
    }

    /**
     * Set up custom Monaco themes
     */
    setupMonacoThemes() {
        // Custom dark theme matching our UI
        monaco.editor.defineTheme('ai-tutorial-dark', {
            base: 'vs-dark',
            inherit: true,
            rules: [
                { token: 'comment', foreground: '6A9955', fontStyle: 'italic' },
                { token: 'keyword', foreground: 'C586C0' },
                { token: 'string', foreground: 'CE9178' },
                { token: 'number', foreground: 'B5CEA8' },
                { token: 'type', foreground: '4EC9B0' },
                { token: 'function', foreground: 'DCDCAA' },
                { token: 'variable', foreground: '9CDCFE' },
                { token: 'class', foreground: '4EC9B0' },
            ],
            colors: {
                'editor.background': '#0f172a',
                'editor.foreground': '#e2e8f0',
                'editorLineNumber.foreground': '#64748b',
                'editorLineNumber.activeForeground': '#94a3b8',
                'editor.selectionBackground': '#334155',
                'editor.lineHighlightBackground': '#1e293b',
                'editorCursor.foreground': '#38bdf8',
                'editorWhitespace.foreground': '#334155',
            }
        });
    }

    /**
     * Initialize the editor
     */
    async initialize() {
        if (this.isReady) return;

        await this.loadMonaco();

        // Create editor
        this.editor = monaco.editor.create(this.container, {
            value: this.options.value,
            language: this.options.language,
            theme: 'ai-tutorial-dark',
            fontSize: this.options.fontSize,
            fontFamily: "'JetBrains Mono', 'Fira Code', 'Monaco', 'Menlo', monospace",
            minimap: { enabled: this.options.minimap },
            lineNumbers: this.options.lineNumbers ? 'on' : 'off',
            wordWrap: this.options.wordWrap,
            automaticLayout: this.options.automaticLayout,
            readOnly: this.options.readOnly,
            scrollBeyondLastLine: false,
            renderLineHighlight: 'all',
            cursorBlinking: 'smooth',
            cursorSmoothCaretAnimation: 'on',
            smoothScrolling: true,
            padding: { top: 16, bottom: 16 },
            folding: true,
            foldingHighlight: true,
            showFoldingControls: 'mouseover',
            bracketPairColorization: { enabled: true },
            guides: {
                bracketPairs: true,
                indentation: true,
            },
            suggest: {
                showKeywords: true,
                showSnippets: true,
            },
            quickSuggestions: {
                other: true,
                comments: false,
                strings: false
            },
            tabSize: 4,
            insertSpaces: true,
        });

        // Set up event listeners
        this.setupEventListeners();

        // Set up Python-specific features
        if (this.options.language === 'python') {
            this.setupPythonFeatures();
        }

        this.isReady = true;
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Content change listener
        this.editor.onDidChangeModelContent(() => {
            this.options.onChange(this.getValue());
        });

        // Save shortcut (Ctrl+S / Cmd+S)
        this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, () => {
            this.options.onSave(this.getValue());
        });

        // Run shortcut (Ctrl+Enter / Cmd+Enter)
        this.editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, () => {
            if (this.options.onRun) {
                this.options.onRun(this.getValue());
            }
        });
    }

    /**
     * Set up Python-specific features
     */
    setupPythonFeatures() {
        // Register Python completions
        monaco.languages.registerCompletionItemProvider('python', {
            provideCompletionItems: (model, position) => {
                const suggestions = this.getPythonSuggestions(model, position);
                return { suggestions };
            }
        });

        // Register hover provider for documentation
        monaco.languages.registerHoverProvider('python', {
            provideHover: (model, position) => {
                return this.getPythonHover(model, position);
            }
        });
    }

    /**
     * Get Python autocompletion suggestions
     */
    getPythonSuggestions(model, position) {
        const word = model.getWordUntilPosition(position);
        const range = {
            startLineNumber: position.lineNumber,
            endLineNumber: position.lineNumber,
            startColumn: word.startColumn,
            endColumn: word.endColumn
        };

        // Common Python snippets and completions
        const suggestions = [
            // Keywords
            ...['def', 'class', 'if', 'elif', 'else', 'for', 'while', 'try', 'except', 'finally',
                'with', 'as', 'import', 'from', 'return', 'yield', 'raise', 'pass', 'break',
                'continue', 'lambda', 'async', 'await', 'True', 'False', 'None'].map(kw => ({
                label: kw,
                kind: monaco.languages.CompletionItemKind.Keyword,
                insertText: kw,
                range
            })),

            // Built-in functions
            ...['print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
                'type', 'isinstance', 'hasattr', 'getattr', 'setattr', 'open', 'input', 'map',
                'filter', 'zip', 'enumerate', 'sorted', 'reversed', 'sum', 'min', 'max', 'abs',
                'round', 'any', 'all'].map(fn => ({
                label: fn,
                kind: monaco.languages.CompletionItemKind.Function,
                insertText: fn + '($0)',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                range
            })),

            // Snippets
            {
                label: 'def function',
                kind: monaco.languages.CompletionItemKind.Snippet,
                insertText: 'def ${1:function_name}(${2:args}):\n    ${3:pass}',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Define a new function',
                range
            },
            {
                label: 'class',
                kind: monaco.languages.CompletionItemKind.Snippet,
                insertText: 'class ${1:ClassName}:\n    def __init__(self${2:, args}):\n        ${3:pass}',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Define a new class',
                range
            },
            {
                label: 'for loop',
                kind: monaco.languages.CompletionItemKind.Snippet,
                insertText: 'for ${1:item} in ${2:items}:\n    ${3:pass}',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'For loop',
                range
            },
            {
                label: 'try except',
                kind: monaco.languages.CompletionItemKind.Snippet,
                insertText: 'try:\n    ${1:pass}\nexcept ${2:Exception} as e:\n    ${3:print(e)}',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Try-except block',
                range
            },
            {
                label: 'with open',
                kind: monaco.languages.CompletionItemKind.Snippet,
                insertText: 'with open("${1:filename}", "${2:r}") as f:\n    ${3:content = f.read()}',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Open file with context manager',
                range
            },
            {
                label: 'async def',
                kind: monaco.languages.CompletionItemKind.Snippet,
                insertText: 'async def ${1:function_name}(${2:args}):\n    ${3:pass}',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Define an async function',
                range
            },

            // AI Tutorial specific helpers
            {
                label: 'print_section',
                kind: monaco.languages.CompletionItemKind.Function,
                insertText: 'print_section("${1:Title}", "${2:content}")',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Print a formatted section header',
                range
            },
            {
                label: 'print_success',
                kind: monaco.languages.CompletionItemKind.Function,
                insertText: 'print_success("${1:message}")',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Print a success message with ✅',
                range
            },
            {
                label: 'print_error',
                kind: monaco.languages.CompletionItemKind.Function,
                insertText: 'print_error("${1:message}")',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Print an error message with ❌',
                range
            },
            {
                label: 'print_info',
                kind: monaco.languages.CompletionItemKind.Function,
                insertText: 'print_info("${1:message}")',
                insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
                documentation: 'Print an info message with ℹ️',
                range
            },
        ];

        return suggestions;
    }

    /**
     * Get hover documentation
     */
    getPythonHover(model, position) {
        const word = model.getWordAtPosition(position);
        if (!word) return null;

        const docs = {
            'print': 'print(*objects, sep=" ", end="\\n", file=sys.stdout, flush=False)\n\nPrint objects to the text stream file.',
            'len': 'len(s)\n\nReturn the length (the number of items) of an object.',
            'range': 'range(stop) or range(start, stop[, step])\n\nReturn an immutable sequence of numbers.',
            'print_section': 'print_section(title, content="")\n\nPrint a formatted section header for lab outputs.',
            'print_success': 'print_success(message)\n\nPrint a success message with a ✅ prefix.',
            'print_error': 'print_error(message)\n\nPrint an error message with a ❌ prefix.',
            'print_info': 'print_info(message)\n\nPrint an info message with an ℹ️ prefix.',
        };

        if (docs[word.word]) {
            return {
                range: new monaco.Range(
                    position.lineNumber,
                    word.startColumn,
                    position.lineNumber,
                    word.endColumn
                ),
                contents: [
                    { value: '```python\n' + docs[word.word] + '\n```' }
                ]
            };
        }

        return null;
    }

    /**
     * Get current editor value
     */
    getValue() {
        return this.editor ? this.editor.getValue() : '';
    }

    /**
     * Set editor value
     */
    setValue(value) {
        if (this.editor) {
            this.editor.setValue(value);
        }
    }

    /**
     * Insert text at cursor position
     */
    insertAtCursor(text) {
        if (this.editor) {
            const selection = this.editor.getSelection();
            const id = { major: 1, minor: 1 };
            const op = {
                identifier: id,
                range: selection,
                text: text,
                forceMoveMarkers: true
            };
            this.editor.executeEdits('insert', [op]);
        }
    }

    /**
     * Focus the editor
     */
    focus() {
        if (this.editor) {
            this.editor.focus();
        }
    }

    /**
     * Set read-only mode
     */
    setReadOnly(readOnly) {
        if (this.editor) {
            this.editor.updateOptions({ readOnly });
        }
    }

    /**
     * Get selected text
     */
    getSelectedText() {
        if (this.editor) {
            return this.editor.getModel().getValueInRange(this.editor.getSelection());
        }
        return '';
    }

    /**
     * Add error marker at line
     */
    addErrorMarker(line, message, startColumn = 1, endColumn = 1000) {
        if (this.editor) {
            const model = this.editor.getModel();
            monaco.editor.setModelMarkers(model, 'errors', [{
                startLineNumber: line,
                startColumn: startColumn,
                endLineNumber: line,
                endColumn: endColumn,
                message: message,
                severity: monaco.MarkerSeverity.Error
            }]);
        }
    }

    /**
     * Clear all error markers
     */
    clearMarkers() {
        if (this.editor) {
            const model = this.editor.getModel();
            monaco.editor.setModelMarkers(model, 'errors', []);
        }
    }

    /**
     * Dispose the editor
     */
    dispose() {
        if (this.editor) {
            this.editor.dispose();
            this.editor = null;
            this.isReady = false;
        }
    }

    /**
     * Resize the editor
     */
    layout() {
        if (this.editor) {
            this.editor.layout();
        }
    }
}

// Simple fallback editor for when Monaco isn't loaded
class SimpleCodeEditor {
    constructor(container, options = {}) {
        this.container = typeof container === 'string' 
            ? document.querySelector(container) 
            : container;
        this.options = options;
        this.textarea = null;
    }

    async initialize() {
        this.container.innerHTML = '';
        
        this.textarea = document.createElement('textarea');
        this.textarea.className = 'w-full h-full bg-slate-900 text-slate-100 font-mono text-sm p-4 resize-none focus:outline-none';
        this.textarea.value = this.options.value || '';
        this.textarea.spellcheck = false;
        
        if (this.options.readOnly) {
            this.textarea.readOnly = true;
        }
        
        this.textarea.addEventListener('input', () => {
            if (this.options.onChange) {
                this.options.onChange(this.getValue());
            }
        });

        // Keyboard shortcuts
        this.textarea.addEventListener('keydown', (e) => {
            // Ctrl+S / Cmd+S
            if ((e.ctrlKey || e.metaKey) && e.key === 's') {
                e.preventDefault();
                if (this.options.onSave) {
                    this.options.onSave(this.getValue());
                }
            }
            // Ctrl+Enter / Cmd+Enter
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                e.preventDefault();
                if (this.options.onRun) {
                    this.options.onRun(this.getValue());
                }
            }
            // Tab handling
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = this.textarea.selectionStart;
                const end = this.textarea.selectionEnd;
                this.textarea.value = this.textarea.value.substring(0, start) + '    ' + this.textarea.value.substring(end);
                this.textarea.selectionStart = this.textarea.selectionEnd = start + 4;
            }
        });

        this.container.appendChild(this.textarea);
        this.isReady = true;
    }

    getValue() {
        return this.textarea ? this.textarea.value : '';
    }

    setValue(value) {
        if (this.textarea) {
            this.textarea.value = value;
        }
    }

    focus() {
        if (this.textarea) {
            this.textarea.focus();
        }
    }

    setReadOnly(readOnly) {
        if (this.textarea) {
            this.textarea.readOnly = readOnly;
        }
    }

    insertAtCursor(text) {
        if (this.textarea) {
            const start = this.textarea.selectionStart;
            const end = this.textarea.selectionEnd;
            this.textarea.value = this.textarea.value.substring(0, start) + text + this.textarea.value.substring(end);
            this.textarea.selectionStart = this.textarea.selectionEnd = start + text.length;
        }
    }

    dispose() {
        if (this.textarea) {
            this.container.innerHTML = '';
            this.textarea = null;
        }
    }

    layout() {
        // No-op for simple editor
    }

    addErrorMarker() {
        // No-op for simple editor
    }

    clearMarkers() {
        // No-op for simple editor
    }
}

/**
 * Create an editor - uses Monaco if available, falls back to simple editor
 */
async function createCodeEditor(container, options = {}) {
    // Try Monaco first
    const editor = new CodeEditor(container, options);
    
    try {
        await editor.initialize();
        return editor;
    } catch (error) {
        console.warn('Monaco editor failed to load, using fallback:', error);
        
        // Fall back to simple editor
        const simpleEditor = new SimpleCodeEditor(container, options);
        await simpleEditor.initialize();
        return simpleEditor;
    }
}

// Export
window.CodeEditor = CodeEditor;
window.SimpleCodeEditor = SimpleCodeEditor;
window.createCodeEditor = createCodeEditor;
