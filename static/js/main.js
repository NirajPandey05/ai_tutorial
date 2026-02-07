/**
 * AI Engineering Tutorial - Main JavaScript
 */

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

/**
 * Initialize the application
 */
function initializeApp() {
    // Initialize code highlighting
    if (typeof hljs !== 'undefined') {
        hljs.highlightAll();
    }
    
    // Initialize copy buttons for code blocks
    initializeCopyButtons();
    
    // Load user progress
    loadProgress();
    
    console.log('🚀 AI Engineering Tutorial initialized');
}

/**
 * Add copy buttons to all code blocks
 */
function initializeCopyButtons() {
    document.querySelectorAll('pre code').forEach((codeBlock) => {
        const pre = codeBlock.parentElement;
        pre.style.position = 'relative';
        
        const button = document.createElement('button');
        button.className = 'copy-button absolute top-2 right-2 px-2 py-1 bg-slate-700 hover:bg-slate-600 rounded text-xs';
        button.textContent = 'Copy';
        button.onclick = () => copyToClipboard(codeBlock.textContent, button);
        
        pre.appendChild(button);
    });
}

/**
 * Copy text to clipboard
 */
async function copyToClipboard(text, button) {
    try {
        await navigator.clipboard.writeText(text);
        button.textContent = 'Copied!';
        setTimeout(() => {
            button.textContent = 'Copy';
        }, 2000);
    } catch (err) {
        console.error('Failed to copy:', err);
        button.textContent = 'Failed';
    }
}

/**
 * Load and display user progress
 */
function loadProgress() {
    const progress = JSON.parse(localStorage.getItem('tutorialProgress') || '{}');
    const totalSections = 10; // This will be dynamic later
    const completedSections = Object.keys(progress).filter(k => progress[k].completed).length;
    const percentage = Math.round((completedSections / totalSections) * 100);
    
    // Update progress bar with null checks
    const progressBar = document.querySelector('.progress-bar');
    const progressText = document.querySelector('.progress-text');
    
    if (progressBar) {
        progressBar.style.width = `${percentage}%`;
    }
    if (progressText) {
        progressText.textContent = `${percentage}% complete`;
    }
}

/**
 * Mark a section as complete
 */
function markComplete(section, page) {
    const progress = JSON.parse(localStorage.getItem('tutorialProgress') || '{}');
    const key = `${section}/${page}`;
    
    progress[key] = {
        completed: true,
        completedAt: new Date().toISOString()
    };
    
    localStorage.setItem('tutorialProgress', JSON.stringify(progress));
    loadProgress();
    
    // Notify server (for future sync)
    fetch(`/api/progress/${section}/${page}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ completed: true })
    }).catch(console.error);
}

/**
 * Get API key for a provider
 */
function getApiKey(provider) {
    return localStorage.getItem(`apiKey_${provider}`);
}

/**
 * Get the default provider
 */
function getDefaultProvider() {
    return localStorage.getItem('defaultProvider') || 'openai';
}

/**
 * Make an API call to an LLM provider
 */
async function callLLM(messages, options = {}) {
    const provider = options.provider || getDefaultProvider();
    const apiKey = getApiKey(provider);
    
    if (!apiKey) {
        throw new Error(`No API key configured for ${provider}. Please add it in Settings.`);
    }
    
    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            provider,
            model: options.model || getDefaultModel(provider),
            messages,
            api_key: apiKey,
            temperature: options.temperature ?? 0.7,
            max_tokens: options.maxTokens ?? 1000,
            stream: options.stream ?? false
        })
    });
    
    if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'API call failed');
    }
    
    return response.json();
}

/**
 * Get default model for a provider
 */
function getDefaultModel(provider) {
    const defaults = {
        openai: 'gpt-4o',
        anthropic: 'claude-sonnet-4-20250514',
        google: 'gemini-3-flash',
        xai: 'grok-3'
    };
    return defaults[provider] || defaults.openai;
}

/**
 * Download code as a Python file
 */
async function downloadCode(code, labId, labTitle) {
    try {
        const response = await fetch('/api/export/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                lab_id: labId,
                lab_title: labTitle,
                language: 'python',
                include_comments: true
            })
        });
        
        if (!response.ok) {
            throw new Error('Download failed');
        }
        
        // Get the blob and trigger download
        const blob = await response.blob();
        const filename = labId.replace(/-/g, '_') + '.py';
        
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        a.remove();
        
        showNotification('Code downloaded successfully!', 'success');
    } catch (error) {
        console.error('Download error:', error);
        showNotification('Failed to download code', 'error');
    }
}

/**
 * Create a shareable link for code
 */
async function createShareLink(code, labId, result = null) {
    try {
        const response = await fetch('/api/export/share', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                lab_id: labId,
                result: result
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to create share link');
        }
        
        const data = await response.json();
        const fullUrl = window.location.origin + data.share_url;
        
        // Copy to clipboard
        await navigator.clipboard.writeText(fullUrl);
        showNotification('Share link copied to clipboard!', 'success');
        
        return fullUrl;
    } catch (error) {
        console.error('Share error:', error);
        showNotification('Failed to create share link', 'error');
        return null;
    }
}

/**
 * Generate social share links
 */
async function getSocialShareLinks(code, labId, result = null) {
    try {
        const response = await fetch('/api/export/social-share', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                code: code,
                lab_id: labId,
                result: result
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to generate social links');
        }
        
        return await response.json();
    } catch (error) {
        console.error('Social share error:', error);
        return null;
    }
}

/**
 * Open share dialog with options
 */
function openShareDialog(code, labId, labTitle, result = null) {
    // Create modal
    const modal = document.createElement('div');
    modal.className = 'fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
    modal.innerHTML = `
        <div class="bg-slate-800 rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <div class="flex justify-between items-center mb-4">
                <h3 class="text-xl font-semibold text-white">Share Your Code</h3>
                <button class="close-modal text-slate-400 hover:text-white text-2xl">&times;</button>
            </div>
            
            <div class="space-y-4">
                <button class="share-btn download-btn w-full py-3 px-4 bg-blue-600 hover:bg-blue-700 rounded-lg text-white flex items-center justify-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/>
                    </svg>
                    Download as Python File
                </button>
                
                <button class="share-btn copy-link-btn w-full py-3 px-4 bg-green-600 hover:bg-green-700 rounded-lg text-white flex items-center justify-center gap-2">
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"/>
                    </svg>
                    Copy Share Link
                </button>
                
                <div class="border-t border-slate-700 pt-4">
                    <p class="text-sm text-slate-400 mb-3">Share on social media:</p>
                    <div class="flex justify-center gap-4 social-buttons">
                        <button class="social-btn twitter p-3 bg-slate-700 hover:bg-blue-500 rounded-lg transition-colors" title="Share on Twitter">
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/>
                            </svg>
                        </button>
                        <button class="social-btn linkedin p-3 bg-slate-700 hover:bg-blue-700 rounded-lg transition-colors" title="Share on LinkedIn">
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                            </svg>
                        </button>
                        <button class="social-btn reddit p-3 bg-slate-700 hover:bg-orange-600 rounded-lg transition-colors" title="Share on Reddit">
                            <svg class="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                                <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Close modal handler
    const closeModal = () => modal.remove();
    modal.querySelector('.close-modal').onclick = closeModal;
    modal.onclick = (e) => { if (e.target === modal) closeModal(); };
    
    // Download button
    modal.querySelector('.download-btn').onclick = () => {
        downloadCode(code, labId, labTitle);
        closeModal();
    };
    
    // Copy link button
    modal.querySelector('.copy-link-btn').onclick = async () => {
        await createShareLink(code, labId, result);
        closeModal();
    };
    
    // Social buttons
    getSocialShareLinks(code, labId, result).then(data => {
        if (data && data.social_links) {
            modal.querySelector('.twitter').onclick = () => window.open(data.social_links.twitter, '_blank');
            modal.querySelector('.linkedin').onclick = () => window.open(data.social_links.linkedin, '_blank');
            modal.querySelector('.reddit').onclick = () => window.open(data.social_links.reddit, '_blank');
        }
    });
}

/**
 * Show a notification toast
 */
function showNotification(message, type = 'info') {
    const toast = document.createElement('div');
    const colors = {
        success: 'bg-green-600',
        error: 'bg-red-600',
        info: 'bg-blue-600',
        warning: 'bg-yellow-600'
    };
    
    toast.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg text-white ${colors[type]} shadow-lg z-50 animate-fade-in`;
    toast.textContent = message;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('opacity-0', 'transition-opacity');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Export functions for use in other scripts
window.AITutorial = {
    markComplete,
    getApiKey,
    getDefaultProvider,
    callLLM,
    copyToClipboard,
    downloadCode,
    createShareLink,
    getSocialShareLinks,
    openShareDialog,
    showNotification
};
