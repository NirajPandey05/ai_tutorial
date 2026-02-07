/**
 * Accessibility Utilities for AI Engineering Tutorial
 * Provides ARIA support, keyboard navigation, focus management, and accessibility settings
 */

const AccessibilityManager = {
    // Settings stored in localStorage
    STORAGE_KEY: 'ai_tutorial_a11y',
    
    // Default settings
    defaults: {
        highContrast: false,
        reduceMotion: false,
        largeText: false,
        focusIndicators: true,
        screenReaderMode: false
    },
    
    settings: {},
    
    /**
     * Initialize accessibility features
     */
    init() {
        this.loadSettings();
        this.applySettings();
        this.setupKeyboardNavigation();
        this.setupSkipLinks();
        this.setupFocusManagement();
        this.announcePageChange();
        this.setupLiveRegions();
        
        // Listen for system preference changes
        this.watchSystemPreferences();
        
        console.log('♿ Accessibility features initialized');
    },
    
    /**
     * Load settings from localStorage
     */
    loadSettings() {
        try {
            const stored = localStorage.getItem(this.STORAGE_KEY);
            this.settings = stored ? { ...this.defaults, ...JSON.parse(stored) } : { ...this.defaults };
        } catch (e) {
            this.settings = { ...this.defaults };
        }
        
        // Check system preferences
        if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            this.settings.reduceMotion = true;
        }
        if (window.matchMedia('(prefers-contrast: more)').matches) {
            this.settings.highContrast = true;
        }
    },
    
    /**
     * Save settings to localStorage
     */
    saveSettings() {
        try {
            localStorage.setItem(this.STORAGE_KEY, JSON.stringify(this.settings));
        } catch (e) {
            console.error('Failed to save accessibility settings:', e);
        }
    },
    
    /**
     * Apply current settings to the document
     */
    applySettings() {
        const html = document.documentElement;
        
        // High contrast mode
        html.classList.toggle('high-contrast', this.settings.highContrast);
        
        // Reduce motion
        html.classList.toggle('reduce-motion', this.settings.reduceMotion);
        
        // Large text
        html.classList.toggle('large-text', this.settings.largeText);
        
        // Focus indicators
        html.classList.toggle('focus-visible', this.settings.focusIndicators);
        
        // Screen reader mode
        html.classList.toggle('screen-reader-mode', this.settings.screenReaderMode);
    },
    
    /**
     * Toggle a specific setting
     */
    toggleSetting(key) {
        if (key in this.settings) {
            this.settings[key] = !this.settings[key];
            this.saveSettings();
            this.applySettings();
            this.announce(`${key.replace(/([A-Z])/g, ' $1').trim()} ${this.settings[key] ? 'enabled' : 'disabled'}`);
        }
    },
    
    /**
     * Setup keyboard navigation
     */
    setupKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Skip to main content: Alt + 1
            if (e.altKey && e.key === '1') {
                e.preventDefault();
                this.skipToMain();
            }
            
            // Skip to navigation: Alt + 2
            if (e.altKey && e.key === '2') {
                e.preventDefault();
                this.skipToNav();
            }
            
            // Toggle accessibility panel: Alt + A
            if (e.altKey && e.key === 'a') {
                e.preventDefault();
                this.toggleAccessibilityPanel();
            }
            
            // Escape key handling for modals/dialogs
            if (e.key === 'Escape') {
                this.handleEscape();
            }
            
            // Arrow key navigation in lists
            if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
                this.handleArrowNavigation(e);
            }
        });
        
        // Handle Tab key for focus trapping in modals
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                this.handleTabTrap(e);
            }
        });
    },
    
    /**
     * Setup skip links
     */
    setupSkipLinks() {
        // Create skip links if they don't exist
        if (!document.querySelector('.skip-links')) {
            const skipLinks = document.createElement('div');
            skipLinks.className = 'skip-links sr-only focus-within:not-sr-only';
            skipLinks.innerHTML = `
                <a href="#main-content" class="skip-link">Skip to main content</a>
                <a href="#navigation" class="skip-link">Skip to navigation</a>
                <a href="#search" class="skip-link">Skip to search</a>
            `;
            document.body.insertBefore(skipLinks, document.body.firstChild);
        }
    },
    
    /**
     * Setup focus management
     */
    setupFocusManagement() {
        // Add visible focus indicators
        document.addEventListener('focusin', (e) => {
            if (this.settings.focusIndicators) {
                e.target.classList.add('focus-visible-custom');
            }
        });
        
        document.addEventListener('focusout', (e) => {
            e.target.classList.remove('focus-visible-custom');
        });
        
        // Track focus for returning after modals
        this.lastFocusedElement = null;
    },
    
    /**
     * Skip to main content
     */
    skipToMain() {
        const main = document.querySelector('main, [role="main"], #main-content');
        if (main) {
            main.setAttribute('tabindex', '-1');
            main.focus();
            main.scrollIntoView({ behavior: this.settings.reduceMotion ? 'auto' : 'smooth' });
        }
    },
    
    /**
     * Skip to navigation
     */
    skipToNav() {
        const nav = document.querySelector('nav, [role="navigation"], #navigation');
        if (nav) {
            const firstLink = nav.querySelector('a, button');
            if (firstLink) {
                firstLink.focus();
            }
        }
    },
    
    /**
     * Handle Escape key
     */
    handleEscape() {
        // Close any open modals
        const modal = document.querySelector('[role="dialog"][aria-modal="true"]:not([hidden])');
        if (modal) {
            modal.setAttribute('hidden', '');
            modal.setAttribute('aria-hidden', 'true');
            
            // Return focus to trigger element
            if (this.lastFocusedElement) {
                this.lastFocusedElement.focus();
            }
        }
        
        // Close dropdown menus
        const openMenu = document.querySelector('[aria-expanded="true"]');
        if (openMenu) {
            openMenu.setAttribute('aria-expanded', 'false');
        }
    },
    
    /**
     * Handle arrow key navigation in lists
     */
    handleArrowNavigation(e) {
        const target = e.target;
        
        // Check if we're in a list or menu
        const list = target.closest('[role="menu"], [role="listbox"], .nav-list');
        if (!list) return;
        
        const items = Array.from(list.querySelectorAll('a, button, [role="menuitem"], [role="option"]'));
        const currentIndex = items.indexOf(target);
        
        if (currentIndex === -1) return;
        
        let nextIndex;
        if (e.key === 'ArrowDown') {
            nextIndex = (currentIndex + 1) % items.length;
        } else if (e.key === 'ArrowUp') {
            nextIndex = (currentIndex - 1 + items.length) % items.length;
        }
        
        if (nextIndex !== undefined) {
            e.preventDefault();
            items[nextIndex].focus();
        }
    },
    
    /**
     * Handle tab trapping in modals
     */
    handleTabTrap(e) {
        const modal = document.querySelector('[role="dialog"][aria-modal="true"]:not([hidden])');
        if (!modal) return;
        
        const focusableElements = modal.querySelectorAll(
            'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'
        );
        
        if (focusableElements.length === 0) return;
        
        const firstElement = focusableElements[0];
        const lastElement = focusableElements[focusableElements.length - 1];
        
        if (e.shiftKey && document.activeElement === firstElement) {
            e.preventDefault();
            lastElement.focus();
        } else if (!e.shiftKey && document.activeElement === lastElement) {
            e.preventDefault();
            firstElement.focus();
        }
    },
    
    /**
     * Announce text to screen readers
     */
    announce(message, priority = 'polite') {
        let announcer = document.getElementById('a11y-announcer');
        
        if (!announcer) {
            announcer = document.createElement('div');
            announcer.id = 'a11y-announcer';
            announcer.setAttribute('aria-live', priority);
            announcer.setAttribute('aria-atomic', 'true');
            announcer.className = 'sr-only';
            document.body.appendChild(announcer);
        }
        
        // Clear and set new message
        announcer.textContent = '';
        announcer.setAttribute('aria-live', priority);
        
        // Use setTimeout to ensure the change is announced
        setTimeout(() => {
            announcer.textContent = message;
        }, 100);
    },
    
    /**
     * Announce page changes
     */
    announcePageChange() {
        const title = document.title || 'Page';
        this.announce(`Navigated to ${title}`, 'assertive');
    },
    
    /**
     * Setup live regions for dynamic content
     */
    setupLiveRegions() {
        // Create status region if it doesn't exist
        if (!document.getElementById('status-region')) {
            const statusRegion = document.createElement('div');
            statusRegion.id = 'status-region';
            statusRegion.setAttribute('role', 'status');
            statusRegion.setAttribute('aria-live', 'polite');
            statusRegion.className = 'sr-only';
            document.body.appendChild(statusRegion);
        }
        
        // Create alert region if it doesn't exist
        if (!document.getElementById('alert-region')) {
            const alertRegion = document.createElement('div');
            alertRegion.id = 'alert-region';
            alertRegion.setAttribute('role', 'alert');
            alertRegion.setAttribute('aria-live', 'assertive');
            alertRegion.className = 'sr-only';
            document.body.appendChild(alertRegion);
        }
    },
    
    /**
     * Watch system preferences for changes
     */
    watchSystemPreferences() {
        // Watch for reduced motion preference changes
        window.matchMedia('(prefers-reduced-motion: reduce)').addEventListener('change', (e) => {
            this.settings.reduceMotion = e.matches;
            this.saveSettings();
            this.applySettings();
        });
        
        // Watch for high contrast preference changes
        window.matchMedia('(prefers-contrast: more)').addEventListener('change', (e) => {
            this.settings.highContrast = e.matches;
            this.saveSettings();
            this.applySettings();
        });
    },
    
    /**
     * Toggle accessibility settings panel
     */
    toggleAccessibilityPanel() {
        let panel = document.getElementById('a11y-panel');
        
        if (!panel) {
            panel = this.createAccessibilityPanel();
        }
        
        const isHidden = panel.hasAttribute('hidden');
        
        if (isHidden) {
            this.lastFocusedElement = document.activeElement;
            panel.removeAttribute('hidden');
            panel.removeAttribute('aria-hidden');
            panel.querySelector('button, [tabindex]').focus();
        } else {
            panel.setAttribute('hidden', '');
            panel.setAttribute('aria-hidden', 'true');
            if (this.lastFocusedElement) {
                this.lastFocusedElement.focus();
            }
        }
    },
    
    /**
     * Create accessibility settings panel
     */
    createAccessibilityPanel() {
        const panel = document.createElement('div');
        panel.id = 'a11y-panel';
        panel.setAttribute('role', 'dialog');
        panel.setAttribute('aria-modal', 'true');
        panel.setAttribute('aria-label', 'Accessibility Settings');
        panel.setAttribute('hidden', '');
        panel.className = 'fixed bottom-4 right-4 bg-slate-800 border border-slate-600 rounded-xl p-4 shadow-xl z-50 w-72';
        
        panel.innerHTML = `
            <div class="flex items-center justify-between mb-4">
                <h2 class="text-lg font-semibold text-white">♿ Accessibility</h2>
                <button onclick="AccessibilityManager.toggleAccessibilityPanel()" 
                        class="text-slate-400 hover:text-white"
                        aria-label="Close accessibility panel">✕</button>
            </div>
            
            <div class="space-y-3">
                <label class="flex items-center justify-between cursor-pointer">
                    <span class="text-sm text-slate-300">High Contrast</span>
                    <input type="checkbox" 
                           ${this.settings.highContrast ? 'checked' : ''}
                           onchange="AccessibilityManager.toggleSetting('highContrast')"
                           class="sr-only peer">
                    <div class="w-10 h-6 bg-slate-600 rounded-full peer peer-checked:bg-primary-500 relative after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-4"></div>
                </label>
                
                <label class="flex items-center justify-between cursor-pointer">
                    <span class="text-sm text-slate-300">Reduce Motion</span>
                    <input type="checkbox" 
                           ${this.settings.reduceMotion ? 'checked' : ''}
                           onchange="AccessibilityManager.toggleSetting('reduceMotion')"
                           class="sr-only peer">
                    <div class="w-10 h-6 bg-slate-600 rounded-full peer peer-checked:bg-primary-500 relative after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-4"></div>
                </label>
                
                <label class="flex items-center justify-between cursor-pointer">
                    <span class="text-sm text-slate-300">Large Text</span>
                    <input type="checkbox" 
                           ${this.settings.largeText ? 'checked' : ''}
                           onchange="AccessibilityManager.toggleSetting('largeText')"
                           class="sr-only peer">
                    <div class="w-10 h-6 bg-slate-600 rounded-full peer peer-checked:bg-primary-500 relative after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-4"></div>
                </label>
                
                <label class="flex items-center justify-between cursor-pointer">
                    <span class="text-sm text-slate-300">Focus Indicators</span>
                    <input type="checkbox" 
                           ${this.settings.focusIndicators ? 'checked' : ''}
                           onchange="AccessibilityManager.toggleSetting('focusIndicators')"
                           class="sr-only peer">
                    <div class="w-10 h-6 bg-slate-600 rounded-full peer peer-checked:bg-primary-500 relative after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-5 after:h-5 after:bg-white after:rounded-full after:transition-all peer-checked:after:translate-x-4"></div>
                </label>
            </div>
            
            <div class="mt-4 pt-4 border-t border-slate-700">
                <p class="text-xs text-slate-500">
                    Keyboard shortcuts:<br>
                    Alt+1: Skip to content<br>
                    Alt+2: Skip to navigation<br>
                    Alt+A: Accessibility panel
                </p>
            </div>
        `;
        
        document.body.appendChild(panel);
        return panel;
    },
    
    /**
     * Add ARIA attributes to an element
     */
    addAriaAttributes(element, attributes) {
        for (const [key, value] of Object.entries(attributes)) {
            element.setAttribute(`aria-${key}`, value);
        }
    },
    
    /**
     * Make an element focusable
     */
    makeFocusable(element, tabIndex = 0) {
        element.setAttribute('tabindex', tabIndex);
    },
    
    /**
     * Get current settings
     */
    getSettings() {
        return { ...this.settings };
    }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => AccessibilityManager.init());
} else {
    AccessibilityManager.init();
}

// Export for use in other scripts
if (typeof window !== 'undefined') {
    window.AccessibilityManager = AccessibilityManager;
}
