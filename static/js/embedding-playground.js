/**
 * Embedding Playground - Interactive embedding visualization
 * 
 * This module provides visualization tools for embeddings, including
 * 2D/3D plots, similarity calculations, and clustering.
 */

class EmbeddingPlayground {
    constructor(options = {}) {
        this.onUpdate = options.onUpdate || (() => {});
        this.embeddings = [];
        this.labels = [];
        this.colors = [
            '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
            '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1',
            '#14b8a6', '#f43f5e', '#22c55e', '#eab308', '#a855f7',
        ];
    }

    /**
     * Generate a mock embedding for text (for client-side demos)
     * In production, this would call an embedding API
     */
    generateMockEmbedding(text, dimensions = 384) {
        // Create a deterministic pseudo-random embedding based on text
        const hash = this.hashString(text);
        const embedding = [];
        
        for (let i = 0; i < dimensions; i++) {
            // Generate pseudo-random numbers using hash
            const seed = hash + i * 17;
            embedding.push(this.seededRandom(seed) * 2 - 1);
        }
        
        // Normalize to unit vector
        return this.normalize(embedding);
    }

    /**
     * Simple string hashing function
     */
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash);
    }

    /**
     * Seeded random number generator
     */
    seededRandom(seed) {
        const x = Math.sin(seed) * 10000;
        return x - Math.floor(x);
    }

    /**
     * Normalize a vector to unit length
     */
    normalize(vector) {
        const magnitude = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0));
        return vector.map(v => v / magnitude);
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    cosineSimilarity(a, b) {
        if (a.length !== b.length) {
            throw new Error('Vectors must have the same length');
        }
        
        let dotProduct = 0;
        let normA = 0;
        let normB = 0;
        
        for (let i = 0; i < a.length; i++) {
            dotProduct += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        
        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Calculate Euclidean distance between two vectors
     */
    euclideanDistance(a, b) {
        if (a.length !== b.length) {
            throw new Error('Vectors must have the same length');
        }
        
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += Math.pow(a[i] - b[i], 2);
        }
        return Math.sqrt(sum);
    }

    /**
     * Calculate dot product between two vectors
     */
    dotProduct(a, b) {
        if (a.length !== b.length) {
            throw new Error('Vectors must have the same length');
        }
        
        return a.reduce((sum, val, i) => sum + val * b[i], 0);
    }

    /**
     * Add a text and its embedding to the playground
     */
    addText(text, label = null) {
        const embedding = this.generateMockEmbedding(text);
        this.embeddings.push({
            text,
            label: label || text.substring(0, 30) + (text.length > 30 ? '...' : ''),
            embedding,
            color: this.colors[this.embeddings.length % this.colors.length]
        });
        return this.embeddings.length - 1;
    }

    /**
     * Clear all embeddings
     */
    clear() {
        this.embeddings = [];
    }

    /**
     * Get similarity matrix between all embeddings
     */
    getSimilarityMatrix() {
        const n = this.embeddings.length;
        const matrix = [];
        
        for (let i = 0; i < n; i++) {
            matrix[i] = [];
            for (let j = 0; j < n; j++) {
                matrix[i][j] = this.cosineSimilarity(
                    this.embeddings[i].embedding,
                    this.embeddings[j].embedding
                );
            }
        }
        
        return matrix;
    }

    /**
     * Find most similar embeddings to a query
     */
    findSimilar(queryIndex, topK = 5) {
        const query = this.embeddings[queryIndex];
        if (!query) return [];
        
        const similarities = this.embeddings.map((emb, index) => ({
            index,
            label: emb.label,
            text: emb.text,
            similarity: this.cosineSimilarity(query.embedding, emb.embedding)
        }));
        
        return similarities
            .sort((a, b) => b.similarity - a.similarity)
            .slice(0, topK);
    }

    /**
     * Reduce dimensions using PCA (Principal Component Analysis)
     * Simplified implementation for visualization
     */
    reduceDimensionsPCA(targetDimensions = 2) {
        if (this.embeddings.length === 0) return [];
        
        const data = this.embeddings.map(e => e.embedding);
        const n = data.length;
        const d = data[0].length;
        
        // Center the data
        const mean = new Array(d).fill(0);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < d; j++) {
                mean[j] += data[i][j] / n;
            }
        }
        
        const centered = data.map(row => row.map((val, j) => val - mean[j]));
        
        // Compute covariance matrix (simplified - using random projection for speed)
        // For a proper PCA, we would compute eigenvectors
        // This is an approximation for demo purposes
        const projection = [];
        for (let i = 0; i < targetDimensions; i++) {
            const vector = new Array(d);
            for (let j = 0; j < d; j++) {
                vector[j] = this.seededRandom(i * d + j) * 2 - 1;
            }
            projection.push(this.normalize(vector));
        }
        
        // Project data
        return centered.map((row, index) => {
            const projected = projection.map(p => this.dotProduct(row, p));
            return {
                x: projected[0] || 0,
                y: projected[1] || 0,
                z: projected[2] || 0,
                label: this.embeddings[index].label,
                text: this.embeddings[index].text,
                color: this.embeddings[index].color
            };
        });
    }

    /**
     * Reduce dimensions using t-SNE approximation
     * Simplified implementation for visualization
     */
    reduceDimensionsTSNE(targetDimensions = 2, perplexity = 30) {
        if (this.embeddings.length === 0) return [];
        
        const n = this.embeddings.length;
        
        // Initialize random positions
        const positions = this.embeddings.map((_, i) => ({
            x: this.seededRandom(i * 2) * 10 - 5,
            y: this.seededRandom(i * 2 + 1) * 10 - 5,
            z: this.seededRandom(i * 2 + 2) * 10 - 5,
        }));
        
        // Compute pairwise distances in high-dimensional space
        const distances = [];
        for (let i = 0; i < n; i++) {
            distances[i] = [];
            for (let j = 0; j < n; j++) {
                if (i === j) {
                    distances[i][j] = 0;
                } else {
                    distances[i][j] = this.euclideanDistance(
                        this.embeddings[i].embedding,
                        this.embeddings[j].embedding
                    );
                }
            }
        }
        
        // Simple gradient descent to approximate t-SNE
        const learningRate = 10;
        const iterations = 100;
        
        for (let iter = 0; iter < iterations; iter++) {
            // Compute gradients
            const gradients = positions.map(() => ({ x: 0, y: 0, z: 0 }));
            
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    if (i === j) continue;
                    
                    const dx = positions[i].x - positions[j].x;
                    const dy = positions[i].y - positions[j].y;
                    const dz = positions[i].z - positions[j].z;
                    const lowDist = Math.sqrt(dx * dx + dy * dy + dz * dz) + 0.001;
                    
                    // Attractive force based on high-dimensional similarity
                    const highSim = 1 / (1 + distances[i][j]);
                    // Repulsive force based on low-dimensional distance
                    const lowSim = 1 / (1 + lowDist * lowDist);
                    
                    const force = (highSim - lowSim) / lowDist;
                    
                    gradients[i].x += force * dx;
                    gradients[i].y += force * dy;
                    gradients[i].z += force * dz;
                }
            }
            
            // Update positions
            const lr = learningRate * (1 - iter / iterations);
            for (let i = 0; i < n; i++) {
                positions[i].x += lr * gradients[i].x;
                positions[i].y += lr * gradients[i].y;
                positions[i].z += lr * gradients[i].z;
            }
        }
        
        // Return with metadata
        return positions.map((pos, index) => ({
            x: pos.x,
            y: pos.y,
            z: pos.z,
            label: this.embeddings[index].label,
            text: this.embeddings[index].text,
            color: this.embeddings[index].color
        }));
    }

    /**
     * Reduce dimensions using UMAP approximation
     */
    reduceDimensionsUMAP(targetDimensions = 2) {
        // UMAP is complex - using a simplified approximation
        // similar to t-SNE but with different distance preservation
        return this.reduceDimensionsTSNE(targetDimensions, 15);
    }

    /**
     * Simple K-means clustering
     */
    cluster(k = 3, maxIterations = 100) {
        if (this.embeddings.length < k) return [];
        
        const data = this.embeddings.map(e => e.embedding);
        const n = data.length;
        const d = data[0].length;
        
        // Initialize centroids randomly
        const centroidIndices = [];
        while (centroidIndices.length < k) {
            const idx = Math.floor(this.seededRandom(centroidIndices.length * 7) * n);
            if (!centroidIndices.includes(idx)) {
                centroidIndices.push(idx);
            }
        }
        let centroids = centroidIndices.map(i => [...data[i]]);
        
        // Assignment
        let assignments = new Array(n).fill(0);
        
        for (let iter = 0; iter < maxIterations; iter++) {
            // Assign points to nearest centroid
            const newAssignments = data.map(point => {
                let minDist = Infinity;
                let minIdx = 0;
                centroids.forEach((centroid, idx) => {
                    const dist = this.euclideanDistance(point, centroid);
                    if (dist < minDist) {
                        minDist = dist;
                        minIdx = idx;
                    }
                });
                return minIdx;
            });
            
            // Check for convergence
            if (JSON.stringify(assignments) === JSON.stringify(newAssignments)) {
                break;
            }
            assignments = newAssignments;
            
            // Update centroids
            centroids = centroids.map((_, clusterIdx) => {
                const points = data.filter((_, i) => assignments[i] === clusterIdx);
                if (points.length === 0) return centroids[clusterIdx];
                
                const newCentroid = new Array(d).fill(0);
                points.forEach(point => {
                    point.forEach((val, j) => {
                        newCentroid[j] += val / points.length;
                    });
                });
                return newCentroid;
            });
        }
        
        return assignments.map((cluster, index) => ({
            index,
            cluster,
            label: this.embeddings[index].label,
            text: this.embeddings[index].text
        }));
    }

    /**
     * Get statistics about the embeddings
     */
    getStats() {
        if (this.embeddings.length === 0) {
            return {
                count: 0,
                dimensions: 0,
                avgSimilarity: 0,
                minSimilarity: 0,
                maxSimilarity: 0
            };
        }
        
        const matrix = this.getSimilarityMatrix();
        const similarities = [];
        
        for (let i = 0; i < matrix.length; i++) {
            for (let j = i + 1; j < matrix[i].length; j++) {
                similarities.push(matrix[i][j]);
            }
        }
        
        return {
            count: this.embeddings.length,
            dimensions: this.embeddings[0].embedding.length,
            avgSimilarity: similarities.length > 0 
                ? (similarities.reduce((a, b) => a + b, 0) / similarities.length).toFixed(3)
                : 0,
            minSimilarity: similarities.length > 0 
                ? Math.min(...similarities).toFixed(3)
                : 0,
            maxSimilarity: similarities.length > 0 
                ? Math.max(...similarities).toFixed(3)
                : 0
        };
    }
}

/**
 * 2D Canvas Renderer for embeddings
 */
class EmbeddingRenderer2D {
    constructor(canvas, options = {}) {
        this.canvas = typeof canvas === 'string' ? document.querySelector(canvas) : canvas;
        this.ctx = this.canvas.getContext('2d');
        this.padding = options.padding || 40;
        this.pointRadius = options.pointRadius || 6;
        this.showLabels = options.showLabels !== false;
        this.hoveredPoint = null;
        
        this.setupInteraction();
    }

    setupInteraction() {
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.handleHover(x, y);
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredPoint = null;
            if (this.lastData) {
                this.render(this.lastData);
            }
        });
    }

    handleHover(mouseX, mouseY) {
        if (!this.lastData || !this.transformedPoints) return;
        
        let closest = null;
        let minDist = Infinity;
        
        this.transformedPoints.forEach((point, index) => {
            const dist = Math.sqrt(
                Math.pow(point.screenX - mouseX, 2) + 
                Math.pow(point.screenY - mouseY, 2)
            );
            if (dist < minDist && dist < 20) {
                minDist = dist;
                closest = index;
            }
        });
        
        if (closest !== this.hoveredPoint) {
            this.hoveredPoint = closest;
            this.render(this.lastData);
        }
    }

    render(data) {
        this.lastData = data;
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, width, height);
        
        if (!data || data.length === 0) {
            ctx.fillStyle = '#64748b';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Add some text to visualize embeddings', width / 2, height / 2);
            return;
        }
        
        // Calculate bounds
        const xValues = data.map(d => d.x);
        const yValues = data.map(d => d.y);
        const xMin = Math.min(...xValues);
        const xMax = Math.max(...xValues);
        const yMin = Math.min(...yValues);
        const yMax = Math.max(...yValues);
        
        // Scale functions
        const xScale = (x) => this.padding + ((x - xMin) / (xMax - xMin || 1)) * (width - 2 * this.padding);
        const yScale = (y) => height - this.padding - ((y - yMin) / (yMax - yMin || 1)) * (height - 2 * this.padding);
        
        // Store transformed points for hover detection
        this.transformedPoints = data.map(d => ({
            ...d,
            screenX: xScale(d.x),
            screenY: yScale(d.y)
        }));
        
        // Draw grid
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const x = this.padding + (i / 4) * (width - 2 * this.padding);
            const y = this.padding + (i / 4) * (height - 2 * this.padding);
            
            ctx.beginPath();
            ctx.moveTo(x, this.padding);
            ctx.lineTo(x, height - this.padding);
            ctx.stroke();
            
            ctx.beginPath();
            ctx.moveTo(this.padding, y);
            ctx.lineTo(width - this.padding, y);
            ctx.stroke();
        }
        
        // Draw points
        this.transformedPoints.forEach((point, index) => {
            const isHovered = index === this.hoveredPoint;
            const radius = isHovered ? this.pointRadius * 1.5 : this.pointRadius;
            
            // Glow effect for hovered point
            if (isHovered) {
                ctx.beginPath();
                ctx.arc(point.screenX, point.screenY, radius + 8, 0, Math.PI * 2);
                ctx.fillStyle = point.color + '40';
                ctx.fill();
            }
            
            // Point
            ctx.beginPath();
            ctx.arc(point.screenX, point.screenY, radius, 0, Math.PI * 2);
            ctx.fillStyle = point.color;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = isHovered ? 2 : 1;
            ctx.stroke();
            
            // Label
            if (this.showLabels || isHovered) {
                ctx.fillStyle = isHovered ? '#fff' : '#94a3b8';
                ctx.font = isHovered ? 'bold 12px sans-serif' : '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.fillText(point.label, point.screenX, point.screenY - radius - 5);
            }
        });
        
        // Draw tooltip for hovered point
        if (this.hoveredPoint !== null) {
            const point = this.transformedPoints[this.hoveredPoint];
            this.drawTooltip(point);
        }
    }

    drawTooltip(point) {
        const ctx = this.ctx;
        const text = point.text;
        const maxWidth = 200;
        
        ctx.font = '12px sans-serif';
        const lines = this.wrapText(text, maxWidth);
        const lineHeight = 16;
        const padding = 8;
        const boxWidth = Math.min(maxWidth + padding * 2, ctx.measureText(text).width + padding * 2);
        const boxHeight = lines.length * lineHeight + padding * 2;
        
        let x = point.screenX + 15;
        let y = point.screenY - boxHeight / 2;
        
        // Keep tooltip in bounds
        if (x + boxWidth > this.canvas.width) x = point.screenX - boxWidth - 15;
        if (y < 0) y = 5;
        if (y + boxHeight > this.canvas.height) y = this.canvas.height - boxHeight - 5;
        
        // Background
        ctx.fillStyle = '#1e293b';
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(x, y, boxWidth, boxHeight, 4);
        ctx.fill();
        ctx.stroke();
        
        // Text
        ctx.fillStyle = '#e2e8f0';
        lines.forEach((line, i) => {
            ctx.fillText(line, x + padding, y + padding + (i + 1) * lineHeight - 4);
        });
    }

    wrapText(text, maxWidth) {
        const ctx = this.ctx;
        const words = text.split(' ');
        const lines = [];
        let currentLine = '';
        
        words.forEach(word => {
            const testLine = currentLine + (currentLine ? ' ' : '') + word;
            if (ctx.measureText(testLine).width > maxWidth && currentLine) {
                lines.push(currentLine);
                currentLine = word;
            } else {
                currentLine = testLine;
            }
        });
        
        if (currentLine) lines.push(currentLine);
        return lines.slice(0, 3); // Limit to 3 lines
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        if (this.lastData) {
            this.render(this.lastData);
        }
    }
}

/**
 * Similarity Matrix Renderer
 */
class SimilarityMatrixRenderer {
    constructor(canvas, options = {}) {
        this.canvas = typeof canvas === 'string' ? document.querySelector(canvas) : canvas;
        this.ctx = this.canvas.getContext('2d');
        this.cellSize = options.cellSize || 40;
        this.labelWidth = options.labelWidth || 100;
    }

    render(matrix, labels) {
        const ctx = this.ctx;
        const n = matrix.length;
        
        const width = this.labelWidth + n * this.cellSize;
        const height = this.labelWidth + n * this.cellSize;
        
        this.canvas.width = Math.max(width, 200);
        this.canvas.height = Math.max(height, 200);
        
        // Clear
        ctx.fillStyle = '#0f172a';
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (n === 0) {
            ctx.fillStyle = '#64748b';
            ctx.font = '14px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Add embeddings to see similarity matrix', this.canvas.width / 2, this.canvas.height / 2);
            return;
        }
        
        // Draw cells
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                const value = matrix[i][j];
                const x = this.labelWidth + j * this.cellSize;
                const y = this.labelWidth + i * this.cellSize;
                
                // Color based on similarity (green = similar, red = different)
                const hue = value * 120; // 0 = red, 120 = green
                ctx.fillStyle = `hsl(${hue}, 70%, ${30 + value * 30}%)`;
                ctx.fillRect(x, y, this.cellSize - 1, this.cellSize - 1);
                
                // Value text
                ctx.fillStyle = value > 0.5 ? '#000' : '#fff';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(value.toFixed(2), x + this.cellSize / 2, y + this.cellSize / 2);
            }
        }
        
        // Draw labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px sans-serif';
        
        // Row labels
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        labels.forEach((label, i) => {
            const y = this.labelWidth + i * this.cellSize + this.cellSize / 2;
            const truncated = label.length > 12 ? label.substring(0, 12) + '...' : label;
            ctx.fillText(truncated, this.labelWidth - 5, y);
        });
        
        // Column labels (rotated)
        ctx.save();
        labels.forEach((label, i) => {
            const x = this.labelWidth + i * this.cellSize + this.cellSize / 2;
            ctx.save();
            ctx.translate(x, this.labelWidth - 5);
            ctx.rotate(-Math.PI / 4);
            ctx.textAlign = 'right';
            const truncated = label.length > 12 ? label.substring(0, 12) + '...' : label;
            ctx.fillText(truncated, 0, 0);
            ctx.restore();
        });
        ctx.restore();
    }
}

// Export
window.EmbeddingPlayground = EmbeddingPlayground;
window.EmbeddingRenderer2D = EmbeddingRenderer2D;
window.SimilarityMatrixRenderer = SimilarityMatrixRenderer;
