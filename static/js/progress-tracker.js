/**
 * Progress Tracker Module
 * Handles all progress tracking, persistence, and synchronization
 */

const ProgressTracker = (function() {
    // Storage keys
    const STORAGE_KEYS = {
        PROGRESS: 'ai_tutorial_progress',
        STREAK: 'ai_tutorial_streak',
        LAST_VISIT: 'ai_tutorial_last_visit',
        ACHIEVEMENTS: 'ai_tutorial_achievements',
        PATH_PROGRESS: 'ai_tutorial_path_progress',
        SETTINGS: 'ai_tutorial_settings',
        SYNC_QUEUE: 'ai_tutorial_sync_queue'
    };

    // Progress data structure
    let progressData = {
        version: 1,
        completedItems: {},      // { itemId: { completedAt, type, moduleId, sectionId } }
        labResults: {},          // { labId: { completedAt, output, score } }
        quizScores: {},          // { quizId: { score, totalQuestions, attempts, lastAttempt } }
        timeSpent: {},           // { pageId: totalSeconds }
        bookmarks: [],           // [{ pageId, title, addedAt }]
        notes: {},               // { pageId: [{ text, createdAt }] }
        learningPaths: {},       // { pathId: { startedAt, currentModule, progress } }
        streak: {
            current: 0,
            longest: 0,
            lastActiveDate: null
        },
        achievements: [],        // [{ id, unlockedAt }]
        lastUpdated: null
    };

    // Achievement definitions
    const ACHIEVEMENTS = {
        // Getting Started
        'first-lesson': { name: 'First Steps', description: 'Complete your first lesson', icon: '📖', category: 'getting-started' },
        'first-lab': { name: 'Lab Rat', description: 'Complete your first lab', icon: '🧪', category: 'getting-started' },
        'first-quiz': { name: 'Quiz Whiz', description: 'Complete your first quiz', icon: '📝', category: 'getting-started' },
        'first-api-call': { name: 'Hello, AI!', description: 'Make your first LLM API call', icon: '🤖', category: 'getting-started' },
        
        // Progress Milestones
        '10-lessons': { name: 'Knowledge Seeker', description: 'Complete 10 lessons', icon: '📚', category: 'progress' },
        '25-lessons': { name: 'Dedicated Learner', description: 'Complete 25 lessons', icon: '🎓', category: 'progress' },
        '50-lessons': { name: 'Scholar', description: 'Complete 50 lessons', icon: '👨‍🎓', category: 'progress' },
        '5-labs': { name: 'Lab Assistant', description: 'Complete 5 labs', icon: '🔬', category: 'progress' },
        '10-labs': { name: 'Lab Expert', description: 'Complete 10 labs', icon: '⚗️', category: 'progress' },
        
        // Streak Achievements
        'streak-3': { name: 'Getting Consistent', description: '3-day learning streak', icon: '🔥', category: 'streak' },
        'streak-7': { name: 'Week Warrior', description: '7-day learning streak', icon: '💪', category: 'streak' },
        'streak-14': { name: 'Two-Week Champion', description: '14-day learning streak', icon: '🏆', category: 'streak' },
        'streak-30': { name: 'Monthly Master', description: '30-day learning streak', icon: '👑', category: 'streak' },
        
        // Module Completions
        'module-llm-fundamentals': { name: 'LLM Foundation', description: 'Complete LLM Fundamentals module', icon: '🧠', category: 'modules' },
        'module-rag': { name: 'RAG Expert', description: 'Complete the RAG module', icon: '📚', category: 'modules' },
        'module-agents': { name: 'Agent Builder', description: 'Complete the Agents module', icon: '🤖', category: 'modules' },
        'module-mcp': { name: 'Protocol Master', description: 'Complete the MCP module', icon: '🔌', category: 'modules' },
        
        // Special Achievements
        'perfect-quiz': { name: 'Perfect Score', description: 'Get 100% on any quiz', icon: '💯', category: 'special' },
        'night-owl': { name: 'Night Owl', description: 'Study after midnight', icon: '🦉', category: 'special' },
        'early-bird': { name: 'Early Bird', description: 'Study before 6 AM', icon: '🐦', category: 'special' },
        'speed-learner': { name: 'Speed Learner', description: 'Complete 5 lessons in one day', icon: '⚡', category: 'special' },
        'completionist': { name: 'Completionist', description: 'Complete all content', icon: '🌟', category: 'special' }
    };

    /**
     * Initialize the progress tracker
     */
    function init() {
        loadFromStorage();
        updateStreak();
        checkTimeBasedAchievements();
        setupAutoSave();
        setupVisibilityTracking();
        
        // Dispatch ready event
        window.dispatchEvent(new CustomEvent('progressTrackerReady', { detail: progressData }));
        
        console.log('📊 Progress Tracker initialized');
        return progressData;
    }

    /**
     * Load progress data from localStorage
     */
    function loadFromStorage() {
        try {
            const stored = localStorage.getItem(STORAGE_KEYS.PROGRESS);
            if (stored) {
                const parsed = JSON.parse(stored);
                // Merge with default structure (for backwards compatibility)
                progressData = { ...progressData, ...parsed };
            }
        } catch (e) {
            console.error('Error loading progress:', e);
        }
    }

    /**
     * Save progress data to localStorage
     */
    function saveToStorage() {
        try {
            progressData.lastUpdated = new Date().toISOString();
            localStorage.setItem(STORAGE_KEYS.PROGRESS, JSON.stringify(progressData));
            
            // Queue for server sync if online
            queueForSync();
        } catch (e) {
            console.error('Error saving progress:', e);
        }
    }

    /**
     * Setup auto-save on changes
     */
    function setupAutoSave() {
        // Save periodically
        setInterval(saveToStorage, 30000); // Every 30 seconds
        
        // Save before page unload
        window.addEventListener('beforeunload', saveToStorage);
    }

    /**
     * Track visibility for time spent
     */
    function setupVisibilityTracking() {
        let startTime = Date.now();
        let currentPage = getCurrentPageId();
        
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                // User left, record time
                if (currentPage) {
                    const elapsed = Math.floor((Date.now() - startTime) / 1000);
                    addTimeSpent(currentPage, elapsed);
                }
            } else {
                // User returned
                startTime = Date.now();
            }
        });
    }

    /**
     * Get current page ID from URL
     */
    function getCurrentPageId() {
        const path = window.location.pathname;
        const match = path.match(/\/learn\/([^/]+)\/([^/]+)\/([^/]+)/);
        if (match) {
            return match[3]; // page ID
        }
        return null;
    }

    /**
     * Mark an item as complete
     */
    function markComplete(itemId, type = 'lesson', moduleId = null, sectionId = null) {
        if (!progressData.completedItems[itemId]) {
            progressData.completedItems[itemId] = {
                completedAt: new Date().toISOString(),
                type,
                moduleId,
                sectionId
            };
            
            saveToStorage();
            updateStreak();
            checkAchievements();
            
            // Dispatch event for item completion
            window.dispatchEvent(new CustomEvent('itemCompleted', { 
                detail: { itemId, type, moduleId, sectionId }
            }));
            
            // Dispatch general progress-updated event for path tracking
            window.dispatchEvent(new CustomEvent('progress-updated', {
                detail: { itemId, type, moduleId, action: 'complete' }
            }));
            
            // Show completion toast
            showCompletionToast(type);
            
            return true;
        }
        return false;
    }

    /**
     * Check if an item is complete
     */
    function isComplete(itemId) {
        return !!progressData.completedItems[itemId];
    }

    /**
     * Mark an item as incomplete (undo completion)
     */
    function markIncomplete(itemId) {
        if (progressData.completedItems[itemId]) {
            delete progressData.completedItems[itemId];
            saveToStorage();
            
            // Dispatch progress-updated event
            window.dispatchEvent(new CustomEvent('progress-updated', {
                detail: { itemId, action: 'incomplete' }
            }));
            
            return true;
        }
        return false;
    }

    /**
     * Save lab result
     */
    function saveLabResult(labId, result, score = null) {
        progressData.labResults[labId] = {
            completedAt: new Date().toISOString(),
            output: result,
            score
        };
        
        markComplete(labId, 'lab');
        saveToStorage();
    }

    /**
     * Save quiz score
     */
    function saveQuizScore(quizId, score, totalQuestions) {
        const existing = progressData.quizScores[quizId] || { attempts: 0 };
        
        progressData.quizScores[quizId] = {
            score,
            totalQuestions,
            percentage: Math.round((score / totalQuestions) * 100),
            attempts: existing.attempts + 1,
            lastAttempt: new Date().toISOString(),
            bestScore: Math.max(score, existing.bestScore || 0)
        };
        
        // Check for perfect score achievement
        if (score === totalQuestions) {
            unlockAchievement('perfect-quiz');
        }
        
        markComplete(quizId, 'quiz');
        saveToStorage();
    }

    /**
     * Add time spent on a page
     */
    function addTimeSpent(pageId, seconds) {
        progressData.timeSpent[pageId] = (progressData.timeSpent[pageId] || 0) + seconds;
        saveToStorage();
    }

    /**
     * Update learning streak
     */
    function updateStreak() {
        const today = new Date().toDateString();
        const lastActive = progressData.streak.lastActiveDate;
        
        if (lastActive === today) {
            // Already active today
            return;
        }
        
        if (lastActive) {
            const lastDate = new Date(lastActive);
            const todayDate = new Date(today);
            const diffDays = Math.floor((todayDate - lastDate) / (1000 * 60 * 60 * 24));
            
            if (diffDays === 1) {
                // Consecutive day
                progressData.streak.current++;
            } else if (diffDays > 1) {
                // Streak broken
                progressData.streak.current = 1;
            }
        } else {
            // First time
            progressData.streak.current = 1;
        }
        
        // Update longest streak
        if (progressData.streak.current > progressData.streak.longest) {
            progressData.streak.longest = progressData.streak.current;
        }
        
        progressData.streak.lastActiveDate = today;
        
        // Check streak achievements
        checkStreakAchievements();
        
        saveToStorage();
    }

    /**
     * Check streak-based achievements
     */
    function checkStreakAchievements() {
        const streak = progressData.streak.current;
        
        if (streak >= 3) unlockAchievement('streak-3');
        if (streak >= 7) unlockAchievement('streak-7');
        if (streak >= 14) unlockAchievement('streak-14');
        if (streak >= 30) unlockAchievement('streak-30');
    }

    /**
     * Check time-based achievements
     */
    function checkTimeBasedAchievements() {
        const hour = new Date().getHours();
        
        if (hour >= 0 && hour < 5) {
            unlockAchievement('night-owl');
        }
        if (hour >= 5 && hour < 6) {
            unlockAchievement('early-bird');
        }
    }

    /**
     * Check all achievement conditions
     */
    function checkAchievements() {
        const completed = progressData.completedItems;
        const counts = { lesson: 0, lab: 0, quiz: 0 };
        const moduleCompletions = {};
        
        // Count completions
        Object.values(completed).forEach(item => {
            counts[item.type] = (counts[item.type] || 0) + 1;
            
            if (item.moduleId) {
                moduleCompletions[item.moduleId] = (moduleCompletions[item.moduleId] || 0) + 1;
            }
        });
        
        // First completions
        if (counts.lesson >= 1) unlockAchievement('first-lesson');
        if (counts.lab >= 1) unlockAchievement('first-lab');
        if (counts.quiz >= 1) unlockAchievement('first-quiz');
        
        // Progress milestones
        if (counts.lesson >= 10) unlockAchievement('10-lessons');
        if (counts.lesson >= 25) unlockAchievement('25-lessons');
        if (counts.lesson >= 50) unlockAchievement('50-lessons');
        if (counts.lab >= 5) unlockAchievement('5-labs');
        if (counts.lab >= 10) unlockAchievement('10-labs');
        
        // Check for speed learner (5 lessons in one day)
        checkSpeedLearner();
    }

    /**
     * Check speed learner achievement
     */
    function checkSpeedLearner() {
        const today = new Date().toDateString();
        let todayCount = 0;
        
        Object.values(progressData.completedItems).forEach(item => {
            if (item.type === 'lesson' && new Date(item.completedAt).toDateString() === today) {
                todayCount++;
            }
        });
        
        if (todayCount >= 5) {
            unlockAchievement('speed-learner');
        }
    }

    /**
     * Unlock an achievement
     */
    function unlockAchievement(achievementId) {
        if (!ACHIEVEMENTS[achievementId]) return;
        
        const alreadyUnlocked = progressData.achievements.some(a => a.id === achievementId);
        if (alreadyUnlocked) return;
        
        progressData.achievements.push({
            id: achievementId,
            unlockedAt: new Date().toISOString()
        });
        
        saveToStorage();
        
        // Show achievement notification
        const achievement = ACHIEVEMENTS[achievementId];
        showAchievementToast(achievement);
        
        // Dispatch event
        window.dispatchEvent(new CustomEvent('achievementUnlocked', { 
            detail: { id: achievementId, ...achievement }
        }));
    }

    /**
     * Get all achievements with unlock status
     */
    function getAchievements() {
        const unlockedIds = new Set(progressData.achievements.map(a => a.id));
        
        return Object.entries(ACHIEVEMENTS).map(([id, data]) => ({
            id,
            ...data,
            unlocked: unlockedIds.has(id),
            unlockedAt: progressData.achievements.find(a => a.id === id)?.unlockedAt
        }));
    }

    /**
     * Start a learning path
     */
    function startPath(pathId) {
        if (!progressData.learningPaths[pathId]) {
            progressData.learningPaths[pathId] = {
                startedAt: new Date().toISOString(),
                currentModule: 0,
                progress: 0
            };
            saveToStorage();
        }
        return progressData.learningPaths[pathId];
    }

    /**
     * Update path progress
     */
    function updatePathProgress(pathId, moduleIndex, progress) {
        if (progressData.learningPaths[pathId]) {
            progressData.learningPaths[pathId].currentModule = moduleIndex;
            progressData.learningPaths[pathId].progress = progress;
            saveToStorage();
        }
    }

    /**
     * Add a bookmark
     */
    function addBookmark(pageId, title) {
        const exists = progressData.bookmarks.some(b => b.pageId === pageId);
        if (!exists) {
            progressData.bookmarks.push({
                pageId,
                title,
                addedAt: new Date().toISOString()
            });
            saveToStorage();
        }
    }

    /**
     * Remove a bookmark
     */
    function removeBookmark(pageId) {
        progressData.bookmarks = progressData.bookmarks.filter(b => b.pageId !== pageId);
        saveToStorage();
    }

    /**
     * Add a note to a page
     */
    function addNote(pageId, text) {
        if (!progressData.notes[pageId]) {
            progressData.notes[pageId] = [];
        }
        progressData.notes[pageId].push({
            id: Date.now().toString(),
            text,
            createdAt: new Date().toISOString()
        });
        saveToStorage();
    }

    /**
     * Get progress statistics
     */
    function getStats() {
        const completed = progressData.completedItems;
        const counts = { lesson: 0, lab: 0, quiz: 0 };
        
        Object.values(completed).forEach(item => {
            counts[item.type] = (counts[item.type] || 0) + 1;
        });
        
        const totalTimeSpent = Object.values(progressData.timeSpent).reduce((a, b) => a + b, 0);
        
        return {
            completedLessons: counts.lesson,
            completedLabs: counts.lab,
            completedQuizzes: counts.quiz,
            totalCompleted: counts.lesson + counts.lab + counts.quiz,
            currentStreak: progressData.streak.current,
            longestStreak: progressData.streak.longest,
            achievementsUnlocked: progressData.achievements.length,
            totalAchievements: Object.keys(ACHIEVEMENTS).length,
            totalTimeSpent, // in seconds
            bookmarksCount: progressData.bookmarks.length
        };
    }

    /**
     * Get module progress
     */
    function getModuleProgress(moduleId, totalItems) {
        let completed = 0;
        
        Object.values(progressData.completedItems).forEach(item => {
            if (item.moduleId === moduleId) {
                completed++;
            }
        });
        
        return {
            completed,
            total: totalItems,
            percentage: totalItems > 0 ? Math.round((completed / totalItems) * 100) : 0
        };
    }

    /**
     * Queue data for server sync
     */
    function queueForSync() {
        if (navigator.onLine) {
            syncWithServer();
        } else {
            const queue = JSON.parse(localStorage.getItem(STORAGE_KEYS.SYNC_QUEUE) || '[]');
            queue.push({
                timestamp: Date.now(),
                data: progressData
            });
            localStorage.setItem(STORAGE_KEYS.SYNC_QUEUE, JSON.stringify(queue));
        }
    }

    /**
     * Sync with server (when online)
     */
    async function syncWithServer() {
        try {
            // Process any queued updates first
            const queue = JSON.parse(localStorage.getItem(STORAGE_KEYS.SYNC_QUEUE) || '[]');
            
            if (queue.length > 0) {
                // Clear queue
                localStorage.setItem(STORAGE_KEYS.SYNC_QUEUE, '[]');
            }
            
            // Sync current progress
            const response = await fetch('/api/progress/sync', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(progressData)
            });
            
            if (response.ok) {
                const serverData = await response.json();
                // Merge server data if newer
                if (serverData.lastUpdated > progressData.lastUpdated) {
                    progressData = { ...progressData, ...serverData };
                    saveToStorage();
                }
            }
        } catch (e) {
            console.log('Sync failed, will retry later:', e.message);
        }
    }

    /**
     * Reset all progress
     */
    function resetProgress() {
        progressData = {
            version: 1,
            completedItems: {},
            labResults: {},
            quizScores: {},
            timeSpent: {},
            bookmarks: [],
            notes: {},
            learningPaths: {},
            streak: { current: 0, longest: 0, lastActiveDate: null },
            achievements: [],
            lastUpdated: null
        };
        saveToStorage();
        
        window.dispatchEvent(new CustomEvent('progressReset'));
    }

    /**
     * Export progress data
     */
    function exportProgress() {
        return JSON.stringify(progressData, null, 2);
    }

    /**
     * Import progress data
     */
    function importProgress(jsonString) {
        try {
            const imported = JSON.parse(jsonString);
            progressData = { ...progressData, ...imported };
            saveToStorage();
            return true;
        } catch (e) {
            console.error('Import failed:', e);
            return false;
        }
    }

    /**
     * Show completion toast
     */
    function showCompletionToast(type) {
        const messages = {
            lesson: '📖 Lesson completed!',
            lab: '🧪 Lab completed!',
            quiz: '📝 Quiz completed!'
        };
        
        showToast(messages[type] || 'Item completed!', 'success');
    }

    /**
     * Show achievement toast
     */
    function showAchievementToast(achievement) {
        showToast(`${achievement.icon} Achievement Unlocked: ${achievement.name}`, 'achievement');
    }

    /**
     * Generic toast notification
     */
    function showToast(message, type = 'info') {
        // Create toast element
        const toast = document.createElement('div');
        toast.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg shadow-lg transform transition-all duration-300 translate-y-full z-50`;
        
        // Style based on type
        const styles = {
            success: 'bg-green-600 text-white',
            achievement: 'bg-gradient-to-r from-amber-500 to-orange-500 text-white',
            info: 'bg-blue-600 text-white',
            error: 'bg-red-600 text-white'
        };
        
        toast.className += ` ${styles[type] || styles.info}`;
        toast.textContent = message;
        
        document.body.appendChild(toast);
        
        // Animate in
        requestAnimationFrame(() => {
            toast.classList.remove('translate-y-full');
        });
        
        // Remove after delay
        setTimeout(() => {
            toast.classList.add('translate-y-full', 'opacity-0');
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // Listen for online status
    window.addEventListener('online', syncWithServer);

    // Public API
    return {
        init,
        markComplete,
        markIncomplete,
        isComplete,
        saveLabResult,
        saveQuizScore,
        addTimeSpent,
        getStats,
        getModuleProgress,
        getAchievements,
        unlockAchievement,
        startPath,
        updatePathProgress,
        addBookmark,
        removeBookmark,
        addNote,
        resetProgress,
        exportProgress,
        importProgress,
        showToast,
        get data() { return progressData; }
    };
})();

// Initialize on load
document.addEventListener('DOMContentLoaded', () => {
    ProgressTracker.init();
});

// Export for global use
window.ProgressTracker = ProgressTracker;
