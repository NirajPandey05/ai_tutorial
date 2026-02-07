/**
 * Token Visualizer - Real-time tokenization display
 * 
 * This module provides visualization of how text is tokenized by different
 * LLM providers, showing token boundaries, counts, and cost estimates.
 */

class TokenVisualizer {
    constructor(options = {}) {
        this.onUpdate = options.onUpdate || (() => {});
        this.debounceTime = options.debounceTime || 300;
        this.debounceTimer = null;
        
        // Token colors for visualization
        this.tokenColors = [
            '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
            '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
            '#14b8a6', '#f43f5e', '#22c55e', '#eab308', '#a855f7',
        ];
        
        // Provider pricing (per 1M tokens as of January 2026)
        this.pricing = {
            'openai': {
                'gpt-5.2': { input: 1.75, output: 14.00 },
                'gpt-5.1': { input: 1.25, output: 10.00 },
                'gpt-4o': { input: 2.50, output: 10.00 },
                'gpt-4o-mini': { input: 0.15, output: 0.60 },
                'gpt-4.1': { input: 2.00, output: 8.00 },
                'gpt-4.1-mini': { input: 0.40, output: 1.60 },
            },
            'anthropic': {
                'claude-opus-4.5': { input: 5.00, output: 25.00 },
                'claude-sonnet-4.5': { input: 3.00, output: 15.00 },
                'claude-opus-4': { input: 15.00, output: 75.00 },
                'claude-sonnet-4': { input: 3.00, output: 15.00 },
            },
            'google': {
                'gemini-3-pro': { input: 2.00, output: 12.00 },
                'gemini-3-flash': { input: 0.50, output: 3.00 },
                'gemini-2.5-pro': { input: 1.25, output: 10.00 },
                'gemini-2.5-flash': { input: 0.30, output: 2.50 },
            },
            'xai': {
                'grok-4-heavy': { input: 3.00, output: 15.00 },
                'grok-4': { input: 2.00, output: 10.00 },
            },
            'deepseek': {
                'deepseek-v3.2': { input: 0.28, output: 0.42 },
            },
            'zhipu': {
                'glm-4.7': { input: 0.60, output: 2.20 },
                'glm-4.6': { input: 0.55, output: 2.19 },
            }
        };

        // Simple tokenizer approximations (for client-side estimation)
        // Real tokenization would require the actual tokenizer from each provider
        this.tokenizers = {
            'gpt': this.gptTokenize.bind(this),
            'claude': this.claudeTokenize.bind(this),
            'gemini': this.geminiTokenize.bind(this),
            'grok': this.gptTokenize.bind(this), // Similar to GPT
        };
    }

    /**
     * Approximate GPT tokenization (BPE-based)
     * This is a simplified approximation - real tokenization uses tiktoken
     */
    gptTokenize(text) {
        if (!text) return [];
        
        const tokens = [];
        let remaining = text;
        
        // Common patterns that are typically single tokens
        const patterns = [
            /^[ \t]+/,                    // Whitespace
            /^[\n\r]+/,                   // Newlines
            /^[A-Z][a-z]+/,              // Capitalized words
            /^[a-z]+/,                    // Lowercase words
            /^[0-9]+/,                    // Numbers
            /^[^\w\s]+/,                  // Punctuation/symbols
            /^\s/,                        // Single whitespace
            /^./,                         // Single character fallback
        ];
        
        while (remaining.length > 0) {
            let matched = false;
            
            for (const pattern of patterns) {
                const match = remaining.match(pattern);
                if (match) {
                    // Split long words into subword tokens (BPE approximation)
                    const tokenText = match[0];
                    if (tokenText.length > 4 && /^[a-zA-Z]+$/.test(tokenText)) {
                        // Approximate subword splitting for long words
                        const parts = this.splitIntoSubwords(tokenText);
                        tokens.push(...parts);
                    } else {
                        tokens.push(tokenText);
                    }
                    remaining = remaining.slice(match[0].length);
                    matched = true;
                    break;
                }
            }
            
            if (!matched) {
                tokens.push(remaining[0]);
                remaining = remaining.slice(1);
            }
        }
        
        return tokens;
    }

    /**
     * Split word into BPE-like subwords
     */
    splitIntoSubwords(word) {
        const subwords = [];
        const commonSuffixes = ['ing', 'ed', 'er', 'est', 'ly', 'tion', 'ness', 'ment', 'able', 'ible'];
        const commonPrefixes = ['un', 're', 'pre', 'dis', 'mis', 'over', 'under'];
        
        let remaining = word;
        
        // Check for prefixes
        for (const prefix of commonPrefixes) {
            if (remaining.toLowerCase().startsWith(prefix) && remaining.length > prefix.length + 2) {
                subwords.push(remaining.slice(0, prefix.length));
                remaining = remaining.slice(prefix.length);
                break;
            }
        }
        
        // Check for suffixes
        for (const suffix of commonSuffixes) {
            if (remaining.toLowerCase().endsWith(suffix) && remaining.length > suffix.length + 2) {
                subwords.push(remaining.slice(0, -suffix.length));
                subwords.push(remaining.slice(-suffix.length));
                remaining = '';
                break;
            }
        }
        
        if (remaining) {
            // Split remaining into chunks of ~4 characters
            while (remaining.length > 5) {
                subwords.push(remaining.slice(0, 4));
                remaining = remaining.slice(4);
            }
            if (remaining) {
                subwords.push(remaining);
            }
        }
        
        return subwords.length > 0 ? subwords : [word];
    }

    /**
     * Approximate Claude tokenization
     * Claude uses a different tokenizer but similar principles
     */
    claudeTokenize(text) {
        // Claude's tokenizer is similar to GPT but with some differences
        // For approximation, we use similar logic with slight adjustments
        return this.gptTokenize(text);
    }

    /**
     * Approximate Gemini tokenization
     * Gemini uses SentencePiece tokenization
     */
    geminiTokenize(text) {
        if (!text) return [];
        
        const tokens = [];
        // SentencePiece tends to include leading spaces in tokens
        const words = text.split(/(?=\s)|(?<=\s)/);
        
        for (const word of words) {
            if (word.length <= 4) {
                tokens.push(word);
            } else {
                // Split longer words
                const parts = this.splitIntoSubwords(word);
                tokens.push(...parts);
            }
        }
        
        return tokens;
    }

    /**
     * Tokenize text for a specific provider
     */
    tokenize(text, provider = 'openai') {
        const tokenizerKey = provider === 'anthropic' ? 'claude' : 
                            provider === 'google' ? 'gemini' :
                            provider === 'xai' ? 'grok' : 'gpt';
        
        const tokenizer = this.tokenizers[tokenizerKey];
        return tokenizer ? tokenizer(text) : this.gptTokenize(text);
    }

    /**
     * Count tokens for text
     */
    countTokens(text, provider = 'openai') {
        const tokens = this.tokenize(text, provider);
        return tokens.length;
    }

    /**
     * Estimate cost for token count
     */
    estimateCost(tokenCount, provider, model, isOutput = false) {
        const providerPricing = this.pricing[provider];
        if (!providerPricing) return 0;
        
        // Find best matching model
        let modelPricing = null;
        for (const modelKey of Object.keys(providerPricing)) {
            if (model && model.toLowerCase().includes(modelKey.toLowerCase().replace('-', ''))) {
                modelPricing = providerPricing[modelKey];
                break;
            }
        }
        
        // Default to first model if no match
        if (!modelPricing) {
            modelPricing = Object.values(providerPricing)[0];
        }
        
        const pricePerToken = (isOutput ? modelPricing.output : modelPricing.input) / 1000000;
        return tokenCount * pricePerToken;
    }

    /**
     * Get detailed token analysis
     */
    analyze(text, provider = 'openai', model = null) {
        const tokens = this.tokenize(text, provider);
        const tokenCount = tokens.length;
        const charCount = text.length;
        const wordCount = text.split(/\s+/).filter(w => w.length > 0).length;
        
        // Calculate various ratios
        const charsPerToken = tokenCount > 0 ? (charCount / tokenCount).toFixed(2) : 0;
        const wordsPerToken = tokenCount > 0 ? (wordCount / tokenCount).toFixed(2) : 0;
        
        // Cost estimates
        const inputCost = this.estimateCost(tokenCount, provider, model, false);
        const outputCost = this.estimateCost(tokenCount, provider, model, true);
        
        return {
            text,
            tokens,
            tokenCount,
            charCount,
            wordCount,
            charsPerToken: parseFloat(charsPerToken),
            wordsPerToken: parseFloat(wordsPerToken),
            inputCost,
            outputCost,
            provider,
            model
        };
    }

    /**
     * Generate HTML visualization of tokens
     */
    visualize(text, provider = 'openai') {
        const tokens = this.tokenize(text, provider);
        
        let html = '';
        tokens.forEach((token, index) => {
            const color = this.tokenColors[index % this.tokenColors.length];
            const escapedToken = this.escapeHtml(token);
            const displayToken = token.replace(/\n/g, '↵\n').replace(/ /g, '·');
            
            html += `<span class="token" style="background-color: ${color}20; border-color: ${color}; color: ${color};" title="Token ${index + 1}: '${escapedToken}'">${this.escapeHtml(displayToken)}</span>`;
        });
        
        return html;
    }

    /**
     * Escape HTML special characters
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Debounced update for real-time input
     */
    debouncedAnalyze(text, provider, model, callback) {
        if (this.debounceTimer) {
            clearTimeout(this.debounceTimer);
        }
        
        this.debounceTimer = setTimeout(() => {
            const analysis = this.analyze(text, provider, model);
            callback(analysis);
        }, this.debounceTime);
    }

    /**
     * Get all available models for a provider
     */
    getModels(provider) {
        const providerPricing = this.pricing[provider];
        return providerPricing ? Object.keys(providerPricing) : [];
    }

    /**
     * Get all providers
     */
    getProviders() {
        return Object.keys(this.pricing);
    }

    /**
     * Format cost for display
     */
    formatCost(cost) {
        if (cost < 0.0001) {
            return '$0.0000';
        } else if (cost < 0.01) {
            return '$' + cost.toFixed(4);
        } else if (cost < 1) {
            return '$' + cost.toFixed(3);
        } else {
            return '$' + cost.toFixed(2);
        }
    }

    /**
     * Compare tokenization across providers
     */
    compareProviders(text) {
        const providers = this.getProviders();
        const comparisons = {};
        
        for (const provider of providers) {
            const models = this.getModels(provider);
            const defaultModel = models[0];
            comparisons[provider] = this.analyze(text, provider, defaultModel);
        }
        
        return comparisons;
    }

    /**
     * Calculate context window usage
     */
    contextUsage(tokenCount, contextWindow = 128000) {
        const percentage = (tokenCount / contextWindow) * 100;
        return {
            used: tokenCount,
            total: contextWindow,
            percentage: percentage.toFixed(1),
            remaining: contextWindow - tokenCount
        };
    }
}

/**
 * Context window sizes for different models
 */
const ContextWindows = {
    'openai': {
        'gpt-5.2': 400000,
        'gpt-5.1': 400000,
        'gpt-4o': 128000,
        'gpt-4o-mini': 128000,
        'gpt-4.1': 1000000,
        'gpt-4.1-mini': 1000000,
    },
    'anthropic': {
        'claude-opus-4.5': 200000,
        'claude-sonnet-4.5': 200000,
        'claude-opus-4': 200000,
        'claude-sonnet-4': 200000,
    },
    'google': {
        'gemini-3-pro': 1000000,
        'gemini-3-flash': 1000000,
        'gemini-2.5-pro': 1000000,
        'gemini-2.5-flash': 1000000,
    },
    'xai': {
        'grok-4-heavy': 131072,
        'grok-4': 131072,
    },
    'deepseek': {
        'deepseek-v3.2': 131100,
    },
    'zhipu': {
        'glm-4.7': 204800,
        'glm-4.6': 131100,
    }
};

/**
 * Get context window for a model
 */
function getContextWindow(provider, model) {
    const providerWindows = ContextWindows[provider];
    if (!providerWindows) return 128000; // Default
    
    for (const modelKey of Object.keys(providerWindows)) {
        if (model && model.toLowerCase().includes(modelKey.toLowerCase().replace('-', ''))) {
            return providerWindows[modelKey];
        }
    }
    
    return Object.values(providerWindows)[0] || 128000;
}

// Export
window.TokenVisualizer = TokenVisualizer;
window.ContextWindows = ContextWindows;
window.getContextWindow = getContextWindow;
