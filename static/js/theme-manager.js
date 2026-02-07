/**
 * Theme Manager for AI Engineering Tutorial
 * Handles dark/light mode toggle and responsive design utilities
 */

const ThemeManager = {
    STORAGE_KEY: 'ai_tutorial_theme',
    
    themes: {
        dark: {
            bg: 'bg-slate-900',
            text: 'text-slate-100',
            sidebar: 'bg-slate-800',
            card: 'bg-slate-800',
            border: 'border-slate-700'
        },
        light: {
            bg: 'bg-gray-50',
            text: 'text-gray-900',
            sidebar: 'bg-white',
            card: 'bg-white',
            border: 'border-gray-200'
        }
    },
    
    currentTheme: 'dark',
    
    /**
     * Initialize theme manager
     */
    init() {
        this.loadTheme();
        this.applyTheme();
        this.watchSystemPreference();
        this.setupToggle();
        
        console.log('🎨 Theme manager initialized');
    },
    
    /**
     * Load theme from localStorage or default to dark
     */
    loadTheme() {
        const stored = localStorage.getItem(this.STORAGE_KEY);
        
        if (stored) {
            this.currentTheme = stored;
        } else {
            // Default to dark mode for this tutorial
            this.currentTheme = 'dark';
        }
    },
    
    /**
     * Apply current theme to document
     */
    applyTheme() {
        const html = document.documentElement;
        
        if (this.currentTheme === 'dark') {
            html.classList.add('dark');
            html.classList.remove('light');
        } else {
            html.classList.add('light');
            html.classList.remove('dark');
        }
        
        // Update meta theme-color for mobile browsers
        let metaTheme = document.querySelector('meta[name="theme-color"]');
        if (!metaTheme) {
            metaTheme = document.createElement('meta');
            metaTheme.name = 'theme-color';
            document.head.appendChild(metaTheme);
        }
        metaTheme.content = this.currentTheme === 'dark' ? '#0f172a' : '#f8fafc';
    },
    
    /**
     * Toggle between dark and light themes
     */
    toggle() {
        this.currentTheme = this.currentTheme === 'dark' ? 'light' : 'dark';
        localStorage.setItem(this.STORAGE_KEY, this.currentTheme);
        this.applyTheme();
        
        // Announce change for accessibility
        if (window.AccessibilityManager) {
            AccessibilityManager.announce(`${this.currentTheme} mode enabled`);
        }
    },
    
    /**
     * Set specific theme
     */
    setTheme(theme) {
        if (theme === 'dark' || theme === 'light') {
            this.currentTheme = theme;
            localStorage.setItem(this.STORAGE_KEY, this.currentTheme);
            this.applyTheme();
        }
    },
    
    /**
     * Watch for system preference changes
     */
    watchSystemPreference() {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
            if (!localStorage.getItem(this.STORAGE_KEY)) {
                this.currentTheme = e.matches ? 'dark' : 'light';
                this.applyTheme();
            }
        });
    },
    
    /**
     * Setup theme toggle button
     */
    setupToggle() {
        // Create toggle button if it doesn't exist
        const existingToggle = document.getElementById('theme-toggle');
        if (existingToggle) {
            existingToggle.addEventListener('click', () => this.toggle());
        }
    },
    
    /**
     * Get current theme
     */
    getTheme() {
        return this.currentTheme;
    },
    
    /**
     * Check if dark mode
     */
    isDark() {
        return this.currentTheme === 'dark';
    }
};

/**
 * Responsive Design Utilities
 */
const ResponsiveManager = {
    breakpoints: {
        sm: 640,
        md: 768,
        lg: 1024,
        xl: 1280,
        '2xl': 1536
    },
    
    currentBreakpoint: 'lg',
    
    /**
     * Initialize responsive manager
     */
    init() {
        this.updateBreakpoint();
        this.setupResizeListener();
        this.setupMobileMenu();
        
        console.log('📱 Responsive manager initialized');
    },
    
    /**
     * Get current viewport width
     */
    getWidth() {
        return window.innerWidth;
    },
    
    /**
     * Update current breakpoint
     */
    updateBreakpoint() {
        const width = this.getWidth();
        
        if (width < this.breakpoints.sm) {
            this.currentBreakpoint = 'xs';
        } else if (width < this.breakpoints.md) {
            this.currentBreakpoint = 'sm';
        } else if (width < this.breakpoints.lg) {
            this.currentBreakpoint = 'md';
        } else if (width < this.breakpoints.xl) {
            this.currentBreakpoint = 'lg';
        } else if (width < this.breakpoints['2xl']) {
            this.currentBreakpoint = 'xl';
        } else {
            this.currentBreakpoint = '2xl';
        }
        
        // Update body class
        document.body.setAttribute('data-breakpoint', this.currentBreakpoint);
    },
    
    /**
     * Setup resize listener
     */
    setupResizeListener() {
        let resizeTimeout;
        
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.updateBreakpoint();
                document.dispatchEvent(new CustomEvent('breakpoint-change', {
                    detail: { breakpoint: this.currentBreakpoint }
                }));
            }, 100);
        });
    },
    
    /**
     * Check if mobile view
     */
    isMobile() {
        return this.currentBreakpoint === 'xs' || this.currentBreakpoint === 'sm';
    },
    
    /**
     * Check if tablet view
     */
    isTablet() {
        return this.currentBreakpoint === 'md';
    },
    
    /**
     * Check if desktop view
     */
    isDesktop() {
        return this.currentBreakpoint === 'lg' || this.currentBreakpoint === 'xl' || this.currentBreakpoint === '2xl';
    },
    
    /**
     * Setup mobile menu behavior
     */
    setupMobileMenu() {
        // Auto-collapse sidebar on mobile
        document.addEventListener('breakpoint-change', (e) => {
            if (this.isMobile()) {
                // Trigger sidebar close on Alpine.js
                document.body.setAttribute('x-data', '{ sidebarOpen: false }');
            }
        });
    },
    
    /**
     * Get current breakpoint
     */
    getBreakpoint() {
        return this.currentBreakpoint;
    }
};

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        ThemeManager.init();
        ResponsiveManager.init();
    });
} else {
    ThemeManager.init();
    ResponsiveManager.init();
}

// Export for use
if (typeof window !== 'undefined') {
    window.ThemeManager = ThemeManager;
    window.ResponsiveManager = ResponsiveManager;
}
