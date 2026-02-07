/**
 * Quiz Engine - Interactive quiz system for AI Tutorial
 * Supports multiple question types with immediate feedback and scoring
 */

class QuizEngine {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.options = {
            showFeedback: true,
            allowRetry: true,
            shuffleQuestions: false,
            shuffleAnswers: true,
            passingScore: 70,
            ...options
        };
        
        this.questions = [];
        this.currentIndex = 0;
        this.answers = {};
        this.multiSelectSubmitted = {};
        this.orderingSubmitted = {};
        this.fillBlankSubmitted = {};
        this.score = 0;
        this.completed = false;
        this.reviewing = false;
        this.startTime = null;
        this.endTime = null;
    }
    
    /**
     * Load questions into the quiz
     */
    loadQuestions(questions) {
        // Store original questions for retry functionality
        this.originalQuestions = questions;
        
        // Deep copy questions to avoid modifying the original data
        this.questions = questions.map(q => ({
            ...q,
            options: q.options ? [...q.options] : undefined,
            correct: Array.isArray(q.correct) ? [...q.correct] : q.correct
        }));
        
        // Shuffle questions if enabled
        if (this.options.shuffleQuestions) {
            this.questions = this.shuffle(this.questions);
        }
        
        // Shuffle answer options if enabled
        if (this.options.shuffleAnswers) {
            this.questions.forEach(q => {
                if ((q.type === 'multiple-choice' || q.type === 'multiple-select') && q.options) {
                    // Store original correct answer(s) by their text value
                    const correctTexts = q.type === 'multiple-select' 
                        ? q.correct.map(idx => q.options[idx])
                        : [q.options[q.correct]];
                    
                    // Shuffle options
                    q.options = this.shuffle([...q.options]);
                    
                    // Find new indices for correct answers based on text matching
                    if (q.type === 'multiple-choice') {
                        q.correct = q.options.indexOf(correctTexts[0]);
                    } else if (q.type === 'multiple-select') {
                        q.correct = correctTexts.map(text => q.options.indexOf(text));
                    }
                }
            });
        }
        
        this.currentIndex = 0;
        this.answers = {};
        this.multiSelectSubmitted = {};
        this.orderingSubmitted = {};
        this.fillBlankSubmitted = {};
        this.score = 0;
        this.completed = false;
        this.reviewing = false;
        this.startTime = Date.now();
    }
    
    /**
     * Shuffle array using Fisher-Yates algorithm
     */
    shuffle(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }
    
    /**
     * Render the current question
     */
    render() {
        if (!this.container) return;
        
        if (this.completed) {
            this.renderResults();
            return;
        }
        
        const question = this.questions[this.currentIndex];
        // For multiple-select, ordering and fill-blank, answered means they clicked submit
        const answered = this.completed || this.reviewing || 
            (question.type === 'multiple-select' ? this.multiSelectSubmitted[this.currentIndex] : 
             question.type === 'ordering' ? this.orderingSubmitted[this.currentIndex] :
             question.type === 'fill-blank' ? this.fillBlankSubmitted[this.currentIndex] :
             this.answers[this.currentIndex] !== undefined);
        
        this.container.innerHTML = `
            <div class="quiz-container">
                <!-- Progress Bar -->
                <div class="quiz-progress mb-6">
                    <div class="flex justify-between text-sm text-slate-400 mb-2">
                        <span>Question ${this.currentIndex + 1} of ${this.questions.length}</span>
                        <span>${Math.round(((this.currentIndex + 1) / this.questions.length) * 100)}% Complete</span>
                    </div>
                    <div class="h-2 bg-slate-700 rounded-full overflow-hidden">
                        <div class="h-full bg-primary-500 transition-all duration-300" 
                             style="width: ${((this.currentIndex + 1) / this.questions.length) * 100}%"></div>
                    </div>
                </div>
                
                <!-- Question Card -->
                <div class="quiz-question bg-slate-800 rounded-xl p-6 mb-6">
                    <div class="flex items-start gap-4">
                        <span class="quiz-type-badge px-2 py-1 rounded text-xs font-medium ${this.getTypeBadgeClass(question.type)}">
                            ${this.getTypeLabel(question.type)}
                        </span>
                        ${question.difficulty ? `
                        <span class="px-2 py-1 rounded text-xs font-medium ${this.getDifficultyClass(question.difficulty)}">
                            ${question.difficulty}
                        </span>
                        ` : ''}
                    </div>
                    
                    <h3 class="text-xl font-semibold text-white mt-4 mb-6">${question.question}</h3>
                    
                    ${question.code ? `
                    <pre class="bg-slate-900 rounded-lg p-4 mb-6 overflow-x-auto"><code class="language-python">${this.escapeHtml(question.code)}</code></pre>
                    ` : ''}
                    
                    <div class="quiz-options space-y-3">
                        ${this.renderOptions(question, answered)}
                    </div>
                    
                    ${answered && this.options.showFeedback ? this.renderFeedback(question) : ''}
                </div>
                
                <!-- Navigation -->
                <div class="quiz-nav flex justify-between items-center">
                    <button 
                        onclick="quiz.prevQuestion()"
                        class="px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors ${this.currentIndex === 0 ? 'opacity-50 cursor-not-allowed' : ''}"
                        ${this.currentIndex === 0 ? 'disabled' : ''}>
                        ← Previous
                    </button>
                    
                    <div class="flex gap-2">
                        ${this.questions.map((_, i) => {
                            const q = this.questions[i];
                            const isAnswered = q.type === 'multiple-select' ? this.multiSelectSubmitted[i] :
                                             q.type === 'ordering' ? this.orderingSubmitted[i] :
                                             q.type === 'fill-blank' ? this.fillBlankSubmitted[i] :
                                             this.answers[i] !== undefined;
                            
                            return `
                                <button 
                                    onclick="quiz.goToQuestion(${i})"
                                    class="w-8 h-8 rounded-full text-sm font-medium transition-colors
                                        ${i === this.currentIndex ? 'bg-primary-500 text-white' : 
                                          isAnswered ? 
                                            (this.isCorrect(i) ? 'bg-green-500/20 text-green-400 border border-green-500' : 'bg-red-500/20 text-red-400 border border-red-500') : 
                                            'bg-slate-700 text-slate-400 hover:bg-slate-600'}">
                                    ${i + 1}
                                </button>
                            `;
                        }).join('')}
                    </div>
                    
                    ${this.currentIndex === this.questions.length - 1 ? `
                        <button 
                            onclick="quiz.finish()"
                            class="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors font-medium">
                            Finish Quiz
                        </button>
                    ` : `
                        <button 
                            onclick="quiz.nextQuestion()"
                            class="px-4 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg transition-colors">
                            Next →
                        </button>
                    `}
                </div>
            </div>
        `;
        
        // Highlight code blocks
        if (question.code && window.hljs) {
            this.container.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
            });
        }
    }
    
    /**
     * Render answer options based on question type
     */
    renderOptions(question, answered) {
        const userAnswer = this.answers[this.currentIndex];
        
        switch (question.type) {
            case 'multiple-choice':
                return question.options.map((option, i) => {
                    const isSelected = userAnswer === i;
                    const isCorrect = Number(i) === Number(question.correct);
                    const showResult = answered && this.options.showFeedback;
                    
                    let classes = 'quiz-option flex items-center gap-3 p-4 rounded-lg border-2 cursor-pointer transition-all';
                    if (showResult) {
                        if (isCorrect) classes += ' border-green-500 bg-green-500/10';
                        else if (isSelected) classes += ' border-red-500 bg-red-500/10';
                        else classes += ' border-slate-600 opacity-50';
                    } else if (isSelected) {
                        classes += ' border-primary-500 bg-primary-500/10';
                    } else {
                        classes += ' border-slate-600 hover:border-slate-500 hover:bg-slate-700/50';
                    }
                    
                    return `
                        <div class="${classes}" onclick="quiz.selectAnswer(${i})">
                            <div class="w-6 h-6 rounded-full border-2 flex items-center justify-center flex-shrink-0
                                ${isSelected ? 'border-primary-500 bg-primary-500' : 'border-slate-500'}">
                                ${isSelected ? '<span class="text-white text-sm">✓</span>' : ''}
                            </div>
                            <span class="text-slate-200">${option}</span>
                            ${showResult && isCorrect ? '<span class="ml-auto text-green-400">✓ Correct</span>' : ''}
                            ${showResult && isSelected && !isCorrect ? '<span class="ml-auto text-red-400">✗ Incorrect</span>' : ''}
                        </div>
                    `;
                }).join('');
                
            case 'multiple-select':
                return question.options.map((option, i) => {
                    const isSelected = Array.isArray(userAnswer) && userAnswer.includes(i);
                    const correctIndices = question.correct.map(Number);
                    const isCorrect = correctIndices.includes(i);
                    const showResult = answered && this.options.showFeedback;
                    
                    let classes = 'quiz-option flex items-center gap-3 p-4 rounded-lg border-2 cursor-pointer transition-all';
                    if (showResult) {
                        if (isCorrect && isSelected) classes += ' border-green-500 bg-green-500/10';
                        else if (isCorrect && !isSelected) classes += ' border-yellow-500 bg-yellow-500/10';
                        else if (!isCorrect && isSelected) classes += ' border-red-500 bg-red-500/10';
                        else classes += ' border-slate-600 opacity-50';
                    } else if (isSelected) {
                        classes += ' border-primary-500 bg-primary-500/10';
                    } else {
                        classes += ' border-slate-600 hover:border-slate-500 hover:bg-slate-700/50';
                    }
                    
                    return `
                        <div class="${classes}" onclick="quiz.toggleAnswer(${i})">
                            <div class="w-6 h-6 rounded border-2 flex items-center justify-center flex-shrink-0
                                ${isSelected ? 'border-primary-500 bg-primary-500' : 'border-slate-500'}">
                                ${isSelected ? '<span class="text-white text-sm">✓</span>' : ''}
                            </div>
                            <span class="text-slate-200">${option}</span>
                            ${showResult && isCorrect ? '<span class="ml-auto text-green-400">✓</span>' : ''}
                            ${showResult && isSelected && !isCorrect ? '<span class="ml-auto text-red-400">✗</span>' : ''}
                        </div>
                    `;
                }).join('') + `
                    <p class="text-sm text-slate-500 mt-2">Select all that apply</p>
                    ${!answered ? `
                        <button 
                            onclick="quiz.submitMultipleSelect()"
                            class="mt-4 px-4 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg transition-colors"
                            ${!userAnswer || userAnswer.length === 0 ? 'disabled style="opacity: 0.5; cursor: not-allowed;"' : ''}>
                            Submit Answer
                        </button>
                    ` : ''}
                `;
                
            case 'true-false':
                return ['True', 'False'].map((option, i) => {
                    const isSelected = userAnswer === (i === 0);
                    const isCorrect = question.correct === (i === 0);
                    const showResult = answered && this.options.showFeedback;
                    
                    let classes = 'quiz-option flex items-center gap-3 p-4 rounded-lg border-2 cursor-pointer transition-all';
                    if (showResult) {
                        if (isCorrect) classes += ' border-green-500 bg-green-500/10';
                        else if (isSelected) classes += ' border-red-500 bg-red-500/10';
                        else classes += ' border-slate-600 opacity-50';
                    } else if (isSelected) {
                        classes += ' border-primary-500 bg-primary-500/10';
                    } else {
                        classes += ' border-slate-600 hover:border-slate-500 hover:bg-slate-700/50';
                    }
                    
                    return `
                        <div class="${classes}" onclick="quiz.selectAnswer(${i === 0})">
                            <div class="w-6 h-6 rounded-full border-2 flex items-center justify-center flex-shrink-0
                                ${isSelected ? 'border-primary-500 bg-primary-500' : 'border-slate-500'}">
                                ${isSelected ? '<span class="text-white text-sm">✓</span>' : ''}
                            </div>
                            <span class="text-slate-200 font-medium">${option}</span>
                        </div>
                    `;
                }).join('');
                
            case 'fill-blank':
                return `
                    <div class="space-y-4">
                        <input 
                            type="text" 
                            id="fill-blank-input"
                            value="${userAnswer || ''}"
                            placeholder="Type your answer..."
                            class="w-full px-4 py-3 bg-slate-900 border-2 border-slate-600 rounded-lg text-white focus:border-primary-500 focus:outline-none"
                            onkeyup="quiz.updateFillBlank(this.value)"
                            onkeypress="if(event.key === 'Enter') quiz.submitFillBlank()"
                            ${answered ? 'disabled' : ''}>
                        ${!answered ? `
                            <button 
                                onclick="quiz.submitFillBlank()"
                                class="px-4 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg transition-colors">
                                Submit Answer
                            </button>
                        ` : ''}
                    </div>
                `;
                
            case 'ordering':
                const items = userAnswer || question.options.map((_, i) => i);
                const showOrderingResult = answered && this.options.showFeedback;
                
                return `
                    <div class="ordering-container space-y-2" id="ordering-list">
                        ${items.map((itemIndex, position) => {
                            let itemClasses = 'ordering-item flex items-center gap-3 p-4 rounded-lg bg-slate-700';
                            let resultBadge = '';
                            
                            if (showOrderingResult) {
                                const correctIndex = question.correct[position];
                                if (itemIndex === correctIndex) {
                                    itemClasses += ' border-2 border-green-500/50 bg-green-500/10';
                                    resultBadge = '<span class="text-green-400 ml-auto">✓</span>';
                                } else {
                                    itemClasses += ' border-2 border-red-500/50 bg-red-500/10';
                                    resultBadge = '<span class="text-red-400 ml-auto">✗</span>';
                                }
                            } else {
                                itemClasses += ' cursor-move';
                            }
                            
                            return `
                                <div class="${itemClasses}" data-index="${itemIndex}" ${!answered ? 'draggable="true"' : ''}>
                                    <span class="text-slate-500 font-mono">${position + 1}.</span>
                                    <span class="flex-1">${question.options[itemIndex]}</span>
                                    ${!answered ? `
                                        <div class="flex flex-col gap-1">
                                            <button onclick="quiz.moveItem(${position}, -1)" class="p-1 hover:bg-slate-600 rounded" ${position === 0 ? 'disabled' : ''}>↑</button>
                                            <button onclick="quiz.moveItem(${position}, 1)" class="p-1 hover:bg-slate-600 rounded" ${position === items.length - 1 ? 'disabled' : ''}>↓</button>
                                        </div>
                                    ` : resultBadge}
                                </div>
                            `;
                        }).join('')}
                    </div>
                    ${!answered ? `
                        <button 
                            onclick="quiz.submitOrdering()"
                            class="mt-4 px-4 py-2 bg-primary-500 hover:bg-primary-600 rounded-lg transition-colors">
                            Submit Order
                        </button>
                    ` : ''}
                `;
                
            default:
                return '<p class="text-red-400">Unknown question type</p>';
        }
    }
    
    /**
     * Render feedback after answering
     */
    renderFeedback(question) {
        const isCorrect = this.isCorrect(this.currentIndex);
        
        return `
            <div class="quiz-feedback mt-6 p-4 rounded-lg ${isCorrect ? 'bg-green-500/10 border border-green-500/30' : 'bg-red-500/10 border border-red-500/30'}">
                <div class="flex items-center gap-2 mb-2">
                    <span class="text-2xl">${isCorrect ? '✅' : '❌'}</span>
                    <span class="font-semibold ${isCorrect ? 'text-green-400' : 'text-red-400'}">
                        ${isCorrect ? 'Correct!' : 'Incorrect'}
                    </span>
                </div>
                ${question.explanation ? `
                    <p class="text-slate-300">${question.explanation}</p>
                ` : ''}
            </div>
        `;
    }
    
    /**
     * Render final results
     */
    renderResults() {
        this.endTime = Date.now();
        const duration = Math.round((this.endTime - this.startTime) / 1000);
        const minutes = Math.floor(duration / 60);
        const seconds = duration % 60;
        
        const correctCount = this.questions.filter((_, i) => this.isCorrect(i)).length;
        const percentage = Math.round((correctCount / this.questions.length) * 100);
        const passed = percentage >= this.options.passingScore;
        
        this.container.innerHTML = `
            <div class="quiz-results text-center">
                <!-- Score Card -->
                <div class="bg-slate-800 rounded-xl p-8 mb-6">
                    <div class="text-6xl mb-4">${passed ? '🎉' : '📚'}</div>
                    <h2 class="text-3xl font-bold mb-2 ${passed ? 'text-green-400' : 'text-yellow-400'}">
                        ${passed ? 'Quiz Passed!' : 'Keep Learning!'}
                    </h2>
                    
                    <div class="text-7xl font-bold my-6 ${passed ? 'text-green-400' : 'text-yellow-400'}">
                        ${percentage}%
                    </div>
                    
                    <div class="flex justify-center gap-8 text-slate-400">
                        <div>
                            <div class="text-2xl font-bold text-white">${correctCount}/${this.questions.length}</div>
                            <div class="text-sm">Correct</div>
                        </div>
                        <div>
                            <div class="text-2xl font-bold text-white">${minutes}:${seconds.toString().padStart(2, '0')}</div>
                            <div class="text-sm">Time</div>
                        </div>
                        <div>
                            <div class="text-2xl font-bold text-white">${this.options.passingScore}%</div>
                            <div class="text-sm">Passing</div>
                        </div>
                    </div>
                </div>
                
                <!-- Question Review -->
                <div class="bg-slate-800 rounded-xl p-6 mb-6 text-left">
                    <h3 class="text-lg font-semibold mb-4">Question Review</h3>
                    <div class="space-y-3">
                        ${this.questions.map((q, i) => {
                            const correct = this.isCorrect(i);
                            return `
                                <div class="flex items-center gap-3 p-3 rounded-lg ${correct ? 'bg-green-500/10' : 'bg-red-500/10'}">
                                    <span class="${correct ? 'text-green-400' : 'text-red-400'}">${correct ? '✓' : '✗'}</span>
                                    <span class="flex-1 text-slate-300">${q.question.substring(0, 60)}${q.question.length > 60 ? '...' : ''}</span>
                                    <button onclick="quiz.reviewQuestion(${i})" class="text-sm text-primary-400 hover:text-primary-300">Review</button>
                                </div>
                            `;
                        }).join('')}
                    </div>
                </div>
                
                <!-- Actions -->
                <div class="flex justify-center gap-4">
                    ${this.options.allowRetry ? `
                        <button onclick="quiz.retry()" class="px-6 py-3 bg-primary-500 hover:bg-primary-600 rounded-lg font-medium transition-colors">
                            Try Again
                        </button>
                    ` : ''}
                    <button onclick="window.history.back()" class="px-6 py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors">
                        Back to Lesson
                    </button>
                </div>
            </div>
        `;
        
        // Save results to localStorage
        this.saveResults(percentage, correctCount);
    }
    
    /**
     * Check if answer at index is correct
     */
    isCorrect(index) {
        const question = this.questions[index];
        const answer = this.answers[index];
        
        if (answer === undefined || answer === null) return false;
        
        switch (question.type) {
            case 'multiple-choice':
            case 'true-false':
                // Compare as numbers for index-based answers
                return Number(answer) === Number(question.correct);
                
            case 'multiple-select':
                if (!Array.isArray(answer)) return false;
                const correctSet = new Set(question.correct.map(Number));
                const answerSet = new Set(answer.map(Number));
                return correctSet.size === answerSet.size && 
                       [...correctSet].every(a => answerSet.has(a));
                
            case 'fill-blank':
                const correct = Array.isArray(question.correct) ? question.correct : [question.correct];
                return correct.some(c => 
                    c.toLowerCase().trim() === String(answer).toLowerCase().trim()
                );
                
            case 'ordering':
                return JSON.stringify(answer) === JSON.stringify(question.correct);
                
            default:
                // For questions without explicit type, assume single choice
                return Number(answer) === Number(question.correct);
        }
    }
    
    // Answer handlers
    selectAnswer(value) {
        if (this.answers[this.currentIndex] !== undefined && this.options.showFeedback) return;
        this.answers[this.currentIndex] = value;
        this.render();
    }
    
    toggleAnswer(index) {
        const currentQuestion = this.questions[this.currentIndex];
        
        // Don't allow toggling if already submitted for multiple-select
        if (currentQuestion.type === 'multiple-select' && this.multiSelectSubmitted[this.currentIndex]) {
            return;
        }
        
        // For other question types, don't allow changes after answering
        if (currentQuestion.type !== 'multiple-select' && 
            this.answers[this.currentIndex] !== undefined && 
            this.options.showFeedback) return;
            
        let current = this.answers[this.currentIndex] || [];
        if (!Array.isArray(current)) current = [];
        
        if (current.includes(index)) {
            current = current.filter(i => i !== index);
        } else {
            current.push(index);
        }
        
        this.answers[this.currentIndex] = current;
        this.render();
    }
    
    updateFillBlank(value) {
        this.answers[this.currentIndex] = value;
    }
    
    submitMultipleSelect() {
        if (this.answers[this.currentIndex] && this.answers[this.currentIndex].length > 0) {
            this.multiSelectSubmitted[this.currentIndex] = true;
            this.render();
        }
    }
    
    submitFillBlank() {
        const input = document.getElementById('fill-blank-input');
        if (input && input.value.trim()) {
            this.answers[this.currentIndex] = input.value.trim();
            this.fillBlankSubmitted[this.currentIndex] = true;
            this.render();
        }
    }
    
    moveItem(position, direction) {
        const question = this.questions[this.currentIndex];
        let order = [...(this.answers[this.currentIndex] || question.options.map((_, i) => i))];
        
        const newPos = position + direction;
        if (newPos < 0 || newPos >= order.length) return;
        
        [order[position], order[newPos]] = [order[newPos], order[position]];
        this.answers[this.currentIndex] = order;
        this.render();
    }
    
    submitOrdering() {
        const question = this.questions[this.currentIndex];
        if (!this.answers[this.currentIndex]) {
            this.answers[this.currentIndex] = question.options.map((_, i) => i);
        }
        this.orderingSubmitted[this.currentIndex] = true;
        this.render();
    }
    
    // Navigation
    nextQuestion() {
        if (this.currentIndex < this.questions.length - 1) {
            this.currentIndex++;
            this.reviewing = false;
            this.render();
        }
    }
    
    prevQuestion() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.reviewing = false;
            this.render();
        }
    }
    
    goToQuestion(index) {
        if (index >= 0 && index < this.questions.length) {
            this.currentIndex = index;
            this.reviewing = false;
            this.render();
        }
    }
    
    reviewQuestion(index) {
        this.completed = false;
        this.reviewing = true;
        this.currentIndex = index;
        this.render();
    }
    
    finish() {
        this.completed = true;
        this.render();
    }
    
    retry() {
        this.loadQuestions(this.originalQuestions);
        this.render();
    }
    
    // Helpers
    getTypeBadgeClass(type) {
        const classes = {
            'multiple-choice': 'bg-blue-500/20 text-blue-400',
            'multiple-select': 'bg-purple-500/20 text-purple-400',
            'true-false': 'bg-green-500/20 text-green-400',
            'fill-blank': 'bg-yellow-500/20 text-yellow-400',
            'ordering': 'bg-orange-500/20 text-orange-400'
        };
        return classes[type] || 'bg-slate-500/20 text-slate-400';
    }
    
    getTypeLabel(type) {
        const labels = {
            'multiple-choice': 'Single Choice',
            'multiple-select': 'Multiple Select',
            'true-false': 'True/False',
            'fill-blank': 'Fill in the Blank',
            'ordering': 'Put in Order'
        };
        return labels[type] || type;
    }
    
    getDifficultyClass(difficulty) {
        const classes = {
            'easy': 'bg-green-500/20 text-green-400',
            'medium': 'bg-yellow-500/20 text-yellow-400',
            'hard': 'bg-red-500/20 text-red-400'
        };
        return classes[difficulty.toLowerCase()] || 'bg-slate-500/20 text-slate-400';
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    saveResults(percentage, correctCount) {
        const quizId = this.options.quizId || 'unknown';
        const results = JSON.parse(localStorage.getItem('quizResults') || '{}');
        
        if (!results[quizId] || percentage > results[quizId].bestScore) {
            results[quizId] = {
                bestScore: percentage,
                lastAttempt: new Date().toISOString(),
                attempts: (results[quizId]?.attempts || 0) + 1,
                correctCount,
                totalQuestions: this.questions.length
            };
            localStorage.setItem('quizResults', JSON.stringify(results));
        }
    }
}

// Sample quiz data factory
const QuizData = {
    // LLM Fundamentals Quiz
    llmBasics: [
        {
            type: 'multiple-choice',
            question: 'What does LLM stand for?',
            options: [
                'Large Language Model',
                'Linear Learning Machine',
                'Logical Language Module',
                'Large Learning Matrix'
            ],
            correct: 0,
            explanation: 'LLM stands for Large Language Model - neural networks trained on vast amounts of text data to understand and generate human-like text.',
            difficulty: 'Easy'
        },
        {
            type: 'multiple-choice',
            question: 'Which architecture forms the foundation of most modern LLMs?',
            options: [
                'Recurrent Neural Networks (RNN)',
                'Convolutional Neural Networks (CNN)',
                'Transformer architecture',
                'Long Short-Term Memory (LSTM)'
            ],
            correct: 2,
            explanation: 'The Transformer architecture, introduced in the "Attention Is All You Need" paper (2017), is the foundation of modern LLMs like GPT, Claude, and Gemini.',
            difficulty: 'Easy'
        },
        {
            type: 'true-false',
            question: 'Tokens are always the same as words in LLM processing.',
            correct: false,
            explanation: 'Tokens can be words, parts of words (subwords), or even individual characters. Common words might be single tokens, while rare words are split into multiple tokens.',
            difficulty: 'Easy'
        },
        {
            type: 'multiple-select',
            question: 'Which of the following are valid LLM parameters that affect output? (Select all that apply)',
            options: [
                'Temperature',
                'Max tokens',
                'Pixel density',
                'Top-p (nucleus sampling)',
                'Frame rate'
            ],
            correct: [0, 1, 3],
            explanation: 'Temperature, max tokens, and top-p are common LLM parameters. Pixel density and frame rate are related to graphics, not language models.',
            difficulty: 'Medium'
        },
        {
            type: 'fill-blank',
            question: 'The mechanism that allows Transformers to weigh the importance of different parts of the input is called the _______ mechanism.',
            correct: ['attention', 'self-attention', 'self attention'],
            explanation: 'The attention mechanism (specifically self-attention) allows Transformers to consider relationships between all positions in a sequence simultaneously.',
            difficulty: 'Medium'
        }
    ],
    
    // RAG Quiz
    rag: [
        {
            type: 'multiple-choice',
            question: 'What does RAG stand for?',
            options: [
                'Rapid Automated Generation',
                'Retrieval-Augmented Generation',
                'Recursive Answer Generation',
                'Random Access Grammar'
            ],
            correct: 1,
            explanation: 'RAG stands for Retrieval-Augmented Generation - a technique that combines information retrieval with text generation.',
            difficulty: 'Easy'
        },
        {
            type: 'ordering',
            question: 'Put the basic RAG pipeline steps in the correct order:',
            options: [
                'Generate response using LLM',
                'Embed user query',
                'Retrieve relevant documents',
                'Combine context with query'
            ],
            correct: [1, 2, 3, 0],
            explanation: 'The RAG pipeline: 1) Embed the query, 2) Retrieve relevant docs, 3) Combine with query as context, 4) Generate response.',
            difficulty: 'Medium'
        },
        {
            type: 'multiple-choice',
            question: 'What is chunking in the context of RAG?',
            options: [
                'Compressing the LLM model',
                'Splitting documents into smaller pieces for embedding',
                'Grouping similar queries together',
                'Caching frequently used responses'
            ],
            correct: 1,
            explanation: 'Chunking is the process of splitting large documents into smaller, manageable pieces that can be individually embedded and retrieved.',
            difficulty: 'Easy'
        },
        {
            type: 'true-false',
            question: 'In RAG, the vector database stores the actual document text, not embeddings.',
            correct: false,
            explanation: 'Vector databases store embeddings (numerical representations) of documents. They may also store the original text as metadata, but the search is performed on embeddings.',
            difficulty: 'Medium'
        },
        {
            type: 'multiple-select',
            question: 'Which are common chunking strategies? (Select all that apply)',
            options: [
                'Fixed-size chunking',
                'Semantic chunking',
                'Alphabetical chunking',
                'Recursive character splitting',
                'Random chunking'
            ],
            correct: [0, 1, 3],
            explanation: 'Fixed-size, semantic, and recursive character splitting are valid chunking strategies. Alphabetical and random chunking are not meaningful approaches.',
            difficulty: 'Hard'
        }
    ],
    
    // Agents Quiz
    agents: [
        {
            type: 'multiple-choice',
            question: 'What is the primary characteristic that distinguishes AI Agents from simple chatbots?',
            options: [
                'They use larger models',
                'They can take actions and use tools autonomously',
                'They have better memory',
                'They are always connected to the internet'
            ],
            correct: 1,
            explanation: 'AI Agents can autonomously decide when and how to use tools/take actions to accomplish goals, rather than just generating text responses.',
            difficulty: 'Easy'
        },
        {
            type: 'multiple-choice',
            question: 'In the ReAct pattern, what does ReAct stand for?',
            options: [
                'Reactive Action',
                'Reason + Act',
                'Response + Action',
                'Recall + Activity'
            ],
            correct: 1,
            explanation: 'ReAct stands for Reason + Act - a pattern where the agent alternates between reasoning about what to do and taking actions.',
            difficulty: 'Medium'
        },
        {
            type: 'ordering',
            question: 'Put the basic agent loop steps in order:',
            options: [
                'Execute the chosen tool',
                'Observe the result',
                'Reason about what to do next',
                'Receive user input or goal'
            ],
            correct: [3, 2, 0, 1],
            explanation: 'Agent loop: 1) Receive input, 2) Reason about next action, 3) Execute tool, 4) Observe result (then repeat reasoning).',
            difficulty: 'Medium'
        },
        {
            type: 'true-false',
            question: 'Function calling requires the LLM to actually execute the code itself.',
            correct: false,
            explanation: 'The LLM only decides which function to call and with what parameters. The actual execution happens in your application code.',
            difficulty: 'Easy'
        },
        {
            type: 'fill-blank',
            question: 'MCP stands for Model _______ Protocol, which standardizes how AI agents connect to external tools and data sources.',
            correct: ['Context', 'context'],
            explanation: 'MCP (Model Context Protocol) is a standard protocol that enables AI agents to connect to various tools and data sources in a consistent way.',
            difficulty: 'Easy'
        }
    ]
};

// Export for use
window.QuizEngine = QuizEngine;
window.QuizData = QuizData;
