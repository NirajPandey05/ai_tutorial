/**
 * Gamification System - Badges, achievements, streaks, and progress tracking
 */

class GamificationSystem {
    constructor() {
        this.badges = this.loadBadges();
        this.progress = this.loadProgress();
        this.streaks = this.loadStreaks();
    }
    
    // ==================== Badge Definitions ====================
    
    getBadgeDefinitions() {
        return {
            // Getting Started
            'first-lesson': {
                id: 'first-lesson',
                name: 'First Steps',
                description: 'Complete your first lesson',
                icon: '👶',
                category: 'getting-started',
                rarity: 'common',
                condition: (progress) => progress.lessonsCompleted >= 1
            },
            'first-lab': {
                id: 'first-lab',
                name: 'Lab Rat',
                description: 'Complete your first lab exercise',
                icon: '🧪',
                category: 'getting-started',
                rarity: 'common',
                condition: (progress) => progress.labsCompleted >= 1
            },
            'first-quiz': {
                id: 'first-quiz',
                name: 'Quiz Taker',
                description: 'Pass your first quiz',
                icon: '📝',
                category: 'getting-started',
                rarity: 'common',
                condition: (progress) => progress.quizzesPassed >= 1
            },
            'api-configured': {
                id: 'api-configured',
                name: 'Plugged In',
                description: 'Configure your first API key',
                icon: '🔌',
                category: 'getting-started',
                rarity: 'common',
                condition: (progress) => progress.apiKeysConfigured >= 1
            },
            
            // Module Completion
            'llm-fundamentals': {
                id: 'llm-fundamentals',
                name: 'LLM Scholar',
                description: 'Complete the LLM Fundamentals module',
                icon: '🧠',
                category: 'modules',
                rarity: 'uncommon',
                condition: (progress) => progress.modulesCompleted?.includes('llm-fundamentals')
            },
            'embeddings-master': {
                id: 'embeddings-master',
                name: 'Vector Virtuoso',
                description: 'Complete the Embeddings & Vectors module',
                icon: '🔢',
                category: 'modules',
                rarity: 'uncommon',
                condition: (progress) => progress.modulesCompleted?.includes('embeddings-vectors')
            },
            'rag-expert': {
                id: 'rag-expert',
                name: 'RAG Expert',
                description: 'Complete the RAG module',
                icon: '📚',
                category: 'modules',
                rarity: 'rare',
                condition: (progress) => progress.modulesCompleted?.includes('rag')
            },
            'agent-builder': {
                id: 'agent-builder',
                name: 'Agent Builder',
                description: 'Complete the AI Agents module',
                icon: '🤖',
                category: 'modules',
                rarity: 'rare',
                condition: (progress) => progress.modulesCompleted?.includes('agents')
            },
            'mcp-specialist': {
                id: 'mcp-specialist',
                name: 'Protocol Pioneer',
                description: 'Complete the MCP module',
                icon: '🔗',
                category: 'modules',
                rarity: 'rare',
                condition: (progress) => progress.modulesCompleted?.includes('mcp')
            },
            
            // Streaks
            'streak-3': {
                id: 'streak-3',
                name: 'On a Roll',
                description: 'Maintain a 3-day learning streak',
                icon: '🔥',
                category: 'streaks',
                rarity: 'common',
                condition: (progress) => progress.currentStreak >= 3
            },
            'streak-7': {
                id: 'streak-7',
                name: 'Week Warrior',
                description: 'Maintain a 7-day learning streak',
                icon: '⚡',
                category: 'streaks',
                rarity: 'uncommon',
                condition: (progress) => progress.currentStreak >= 7
            },
            'streak-30': {
                id: 'streak-30',
                name: 'Dedicated Learner',
                description: 'Maintain a 30-day learning streak',
                icon: '🏆',
                category: 'streaks',
                rarity: 'rare',
                condition: (progress) => progress.currentStreak >= 30
            },
            
            // Achievements
            'perfect-quiz': {
                id: 'perfect-quiz',
                name: 'Perfect Score',
                description: 'Get 100% on any quiz',
                icon: '💯',
                category: 'achievements',
                rarity: 'uncommon',
                condition: (progress) => progress.perfectQuizzes >= 1
            },
            'challenge-solver': {
                id: 'challenge-solver',
                name: 'Problem Solver',
                description: 'Solve 5 coding challenges',
                icon: '💻',
                category: 'achievements',
                rarity: 'uncommon',
                condition: (progress) => progress.challengesSolved >= 5
            },
            'no-hints': {
                id: 'no-hints',
                name: 'Unaided',
                description: 'Solve a coding challenge without using hints',
                icon: '🎯',
                category: 'achievements',
                rarity: 'uncommon',
                condition: (progress) => progress.challengesSolvedNoHints >= 1
            },
            'speed-demon': {
                id: 'speed-demon',
                name: 'Speed Demon',
                description: 'Complete a challenge in under 2 minutes',
                icon: '⚡',
                category: 'achievements',
                rarity: 'rare',
                condition: (progress) => progress.fastChallenges >= 1
            },
            
            // Mastery
            'all-providers': {
                id: 'all-providers',
                name: 'Provider Pro',
                description: 'Use all 4 LLM providers',
                icon: '🌐',
                category: 'mastery',
                rarity: 'rare',
                condition: (progress) => progress.providersUsed >= 4
            },
            'completionist': {
                id: 'completionist',
                name: 'Completionist',
                description: 'Complete all available modules',
                icon: '🎓',
                category: 'mastery',
                rarity: 'legendary',
                condition: (progress) => progress.allModulesCompleted
            },
            'ai-engineer': {
                id: 'ai-engineer',
                name: 'AI Engineer',
                description: 'Complete the entire learning path',
                icon: '👑',
                category: 'mastery',
                rarity: 'legendary',
                condition: (progress) => progress.learningPathCompleted
            }
        };
    }
    
    // ==================== Progress Tracking ====================
    
    loadProgress() {
        const stored = localStorage.getItem('learningProgress');
        return stored ? JSON.parse(stored) : {
            lessonsCompleted: 0,
            labsCompleted: 0,
            quizzesPassed: 0,
            perfectQuizzes: 0,
            challengesSolved: 0,
            challengesSolvedNoHints: 0,
            fastChallenges: 0,
            apiKeysConfigured: 0,
            providersUsed: 0,
            modulesCompleted: [],
            sectionsCompleted: [],
            currentStreak: 0,
            longestStreak: 0,
            lastActivityDate: null,
            totalTimeSpent: 0,
            allModulesCompleted: false,
            learningPathCompleted: false
        };
    }
    
    saveProgress() {
        localStorage.setItem('learningProgress', JSON.stringify(this.progress));
        this.checkBadges();
    }
    
    updateProgress(updates) {
        Object.assign(this.progress, updates);
        this.updateStreak();
        this.saveProgress();
    }
    
    // ==================== Streak Tracking ====================
    
    loadStreaks() {
        const stored = localStorage.getItem('streakData');
        return stored ? JSON.parse(stored) : {
            currentStreak: 0,
            longestStreak: 0,
            lastActivityDate: null,
            activityHistory: []
        };
    }
    
    saveStreaks() {
        localStorage.setItem('streakData', JSON.stringify(this.streaks));
    }
    
    updateStreak() {
        const today = new Date().toDateString();
        const lastActivity = this.streaks.lastActivityDate;
        
        if (lastActivity === today) {
            // Already counted today
            return;
        }
        
        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        
        if (lastActivity === yesterday.toDateString()) {
            // Continuing streak
            this.streaks.currentStreak++;
        } else if (lastActivity !== today) {
            // Streak broken
            this.streaks.currentStreak = 1;
        }
        
        if (this.streaks.currentStreak > this.streaks.longestStreak) {
            this.streaks.longestStreak = this.streaks.currentStreak;
        }
        
        this.streaks.lastActivityDate = today;
        this.streaks.activityHistory.push({
            date: today,
            timestamp: Date.now()
        });
        
        // Keep only last 90 days of history
        const ninetyDaysAgo = Date.now() - (90 * 24 * 60 * 60 * 1000);
        this.streaks.activityHistory = this.streaks.activityHistory.filter(
            a => a.timestamp > ninetyDaysAgo
        );
        
        this.progress.currentStreak = this.streaks.currentStreak;
        this.progress.longestStreak = this.streaks.longestStreak;
        
        this.saveStreaks();
    }
    
    // ==================== Badge Management ====================
    
    loadBadges() {
        const stored = localStorage.getItem('earnedBadges');
        return stored ? JSON.parse(stored) : {};
    }
    
    saveBadges() {
        localStorage.setItem('earnedBadges', JSON.stringify(this.badges));
    }
    
    checkBadges() {
        const definitions = this.getBadgeDefinitions();
        const newBadges = [];
        
        for (const [id, badge] of Object.entries(definitions)) {
            if (!this.badges[id] && badge.condition(this.progress)) {
                this.badges[id] = {
                    earnedAt: new Date().toISOString(),
                    ...badge
                };
                newBadges.push(badge);
            }
        }
        
        if (newBadges.length > 0) {
            this.saveBadges();
            newBadges.forEach(badge => this.showBadgeNotification(badge));
        }
        
        return newBadges;
    }
    
    showBadgeNotification(badge) {
        const notification = document.createElement('div');
        notification.className = 'fixed bottom-4 right-4 bg-slate-800 border border-primary-500 rounded-xl p-4 shadow-lg z-50 animate-slide-up';
        notification.innerHTML = `
            <div class="flex items-center gap-4">
                <div class="text-4xl">${badge.icon}</div>
                <div>
                    <div class="text-sm text-primary-400 font-medium">🎉 Badge Earned!</div>
                    <div class="text-white font-semibold">${badge.name}</div>
                    <div class="text-sm text-slate-400">${badge.description}</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('animate-fade-out');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
    
    getEarnedBadges() {
        return Object.values(this.badges);
    }
    
    getAllBadges() {
        const definitions = this.getBadgeDefinitions();
        return Object.entries(definitions).map(([id, badge]) => ({
            ...badge,
            earned: !!this.badges[id],
            earnedAt: this.badges[id]?.earnedAt
        }));
    }
    
    getBadgesByCategory() {
        const badges = this.getAllBadges();
        const categories = {};
        
        badges.forEach(badge => {
            if (!categories[badge.category]) {
                categories[badge.category] = [];
            }
            categories[badge.category].push(badge);
        });
        
        return categories;
    }
    
    // ==================== Activity Tracking ====================
    
    recordLessonComplete(lessonId) {
        this.progress.lessonsCompleted++;
        if (!this.progress.lessonsViewed) this.progress.lessonsViewed = [];
        if (!this.progress.lessonsViewed.includes(lessonId)) {
            this.progress.lessonsViewed.push(lessonId);
        }
        this.updateProgress({});
    }
    
    recordLabComplete(labId) {
        this.progress.labsCompleted++;
        if (!this.progress.labsViewed) this.progress.labsViewed = [];
        if (!this.progress.labsViewed.includes(labId)) {
            this.progress.labsViewed.push(labId);
        }
        this.updateProgress({});
    }
    
    recordQuizComplete(quizId, score, perfect) {
        this.progress.quizzesPassed++;
        if (perfect) this.progress.perfectQuizzes++;
        this.updateProgress({});
    }
    
    recordChallengeComplete(challengeId, hintsUsed, timeSeconds) {
        this.progress.challengesSolved++;
        if (hintsUsed === 0) this.progress.challengesSolvedNoHints++;
        if (timeSeconds < 120) this.progress.fastChallenges++;
        this.updateProgress({});
    }
    
    recordModuleComplete(moduleId) {
        if (!this.progress.modulesCompleted.includes(moduleId)) {
            this.progress.modulesCompleted.push(moduleId);
            
            // Check if all modules are complete
            const allModules = ['llm-fundamentals', 'embeddings-vectors', 'rag', 'agents', 'mcp', 'self-hosting', 'fine-tuning', 'advanced-llm', 'multi-agents'];
            if (allModules.every(m => this.progress.modulesCompleted.includes(m))) {
                this.progress.allModulesCompleted = true;
            }
        }
        this.updateProgress({});
    }
    
    recordApiKeyConfigured(provider) {
        if (!this.progress.providersConfigured) this.progress.providersConfigured = [];
        if (!this.progress.providersConfigured.includes(provider)) {
            this.progress.providersConfigured.push(provider);
            this.progress.apiKeysConfigured = this.progress.providersConfigured.length;
            this.progress.providersUsed = this.progress.providersConfigured.length;
        }
        this.updateProgress({});
    }
    
    // ==================== Stats ====================
    
    getStats() {
        return {
            lessonsCompleted: this.progress.lessonsCompleted,
            labsCompleted: this.progress.labsCompleted,
            quizzesPassed: this.progress.quizzesPassed,
            challengesSolved: this.progress.challengesSolved,
            currentStreak: this.streaks.currentStreak,
            longestStreak: this.streaks.longestStreak,
            badgesEarned: Object.keys(this.badges).length,
            totalBadges: Object.keys(this.getBadgeDefinitions()).length,
            modulesCompleted: this.progress.modulesCompleted.length
        };
    }
    
    // ==================== Skill Tree ====================
    
    getSkillTree() {
        return {
            'foundations': {
                name: 'Foundations',
                skills: [
                    { id: 'llm-basics', name: 'LLM Basics', icon: '🧠', unlocked: this.progress.lessonsCompleted >= 1, progress: Math.min(100, this.progress.lessonsCompleted * 20) },
                    { id: 'prompts', name: 'Prompt Engineering', icon: '✍️', unlocked: this.progress.lessonsCompleted >= 3, progress: 0 },
                    { id: 'tokens', name: 'Tokenization', icon: '🔤', unlocked: false, progress: 0 },
                    { id: 'parameters', name: 'Parameters', icon: '⚙️', unlocked: false, progress: 0 }
                ]
            },
            'intermediate': {
                name: 'Intermediate',
                skills: [
                    { id: 'embeddings', name: 'Embeddings', icon: '📊', unlocked: this.progress.modulesCompleted.includes('embeddings-vectors'), progress: 0 },
                    { id: 'vector-dbs', name: 'Vector DBs', icon: '🗄️', unlocked: false, progress: 0 },
                    { id: 'rag-basics', name: 'RAG Basics', icon: '📚', unlocked: false, progress: 0 },
                    { id: 'chunking', name: 'Chunking', icon: '📄', unlocked: false, progress: 0 }
                ]
            },
            'advanced': {
                name: 'Advanced',
                skills: [
                    { id: 'agents', name: 'AI Agents', icon: '🤖', unlocked: this.progress.modulesCompleted.includes('agents'), progress: 0 },
                    { id: 'tools', name: 'Tool Use', icon: '🔧', unlocked: false, progress: 0 },
                    { id: 'mcp', name: 'MCP Protocol', icon: '🔗', unlocked: false, progress: 0 },
                    { id: 'multi-agent', name: 'Multi-Agent', icon: '👥', unlocked: false, progress: 0 }
                ]
            },
            'expert': {
                name: 'Expert',
                skills: [
                    { id: 'fine-tuning', name: 'Fine-tuning', icon: '🎯', unlocked: false, progress: 0 },
                    { id: 'self-hosting', name: 'Self-Hosting', icon: '🏠', unlocked: false, progress: 0 },
                    { id: 'production', name: 'Production', icon: '🚀', unlocked: false, progress: 0 },
                    { id: 'optimization', name: 'Optimization', icon: '⚡', unlocked: false, progress: 0 }
                ]
            }
        };
    }
}

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slide-up {
        from {
            transform: translateY(100%);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes fade-out {
        from {
            opacity: 1;
        }
        to {
            opacity: 0;
        }
    }
    
    .animate-slide-up {
        animation: slide-up 0.3s ease-out;
    }
    
    .animate-fade-out {
        animation: fade-out 0.3s ease-out forwards;
    }
`;
document.head.appendChild(style);

// Export singleton
window.gamification = new GamificationSystem();
