// Unified Vision Feature Manager for DeepEarth Dashboard
// This module provides a consistent vision feature interface for both Geospatial and Embeddings views

class VisionFeatureManager {
    constructor(config) {
        this.imageId = null;
        this.gbifId = null;
        this.container = config.container;
        this.imageElement = config.imageElement;
        this.overlayElement = config.overlayElement;
        this.overlayContainer = config.overlayContainer;
        this.statsContainer = config.statsContainer;
        
        // Feature settings
        this.temporalMode = 'mean';
        this.temporalFrame = 0;
        this.visualization = 'pca1';
        this.colormap = 'plasma';
        this.alpha = 0.7;
        this.isUMAPActive = false;
        this.umapRGBData = null;
        this.aspectRatio = 1;
        
        // Callbacks
        this.onUpdate = config.onUpdate || (() => {});
    }
    
    async loadImage(imageId, gbifId) {
        this.imageId = imageId;
        this.gbifId = gbifId;
        
        // Clear previous state
        this.clearOverlay();
        
        // Extract GBIF ID and image number from imageId (format: gbif_XXXXXXX_taxon_XXXXXXX_img_N)
        const match = imageId.match(/gbif_(\d+)_taxon_\d+_img_(\d+)/);
        let actualGbifId, imageNum;
        
        if (match) {
            actualGbifId = match[1];
            imageNum = match[2];
        } else {
            // Fallback to passed parameters
            actualGbifId = gbifId;
            imageNum = 1;
        }
        
        // Load image using the correct Flask route format
        this.imageElement.src = `/api/image_proxy/${actualGbifId}/${imageNum}`;
        
        // Wait for image to load and get aspect ratio
        await new Promise((resolve, reject) => {
            this.imageElement.onload = () => {
                this.aspectRatio = this.imageElement.naturalWidth / this.imageElement.naturalHeight;
                resolve();
            };
            this.imageElement.onerror = reject;
        });
        
        // Auto-load features if enabled
        if (this.temporalMode || this.visualization) {
            await this.updateVisualization();
        }
    }
    
    clearOverlay() {
        if (this.overlayElement) {
            this.overlayElement.src = '';
            this.overlayElement.style.transform = 'none';
        }
        if (this.overlayContainer) {
            this.overlayContainer.style.display = 'none';
        }
        this.isUMAPActive = false;
        this.umapRGBData = null;
    }
    
    async updateVisualization() {
        if (!this.imageId || this.isUMAPActive) return;
        
        try {
            const params = new URLSearchParams({
                temporal: this.temporalMode,
                frame: this.temporalFrame,
                visualization: this.visualization,
                colormap: this.colormap,
                alpha: this.alpha
            });
            
            const response = await fetch(`/api/features/${this.imageId}/attention?${params}`);
            
            if (!response.ok) {
                console.warn(`No vision features available for ${this.imageId}`);
                this.clearOverlay();
                this.updateStats(null);
                return;
            }
            
            const data = await response.json();
            
            if (data.attention_map) {
                this.overlayElement.src = data.attention_map;
                
                // Apply aspect ratio correction
                if (this.aspectRatio > 1) {
                    this.overlayElement.style.transform = `scaleX(${this.aspectRatio})`;
                } else if (this.aspectRatio < 1) {
                    this.overlayElement.style.transform = `scaleY(${1 / this.aspectRatio})`;
                } else {
                    this.overlayElement.style.transform = 'none';
                }
                
                this.overlayContainer.style.display = 'block';
                this.overlayContainer.style.opacity = '1';
            }
            
            // Update statistics
            this.updateStats(data.stats);
            
            // Callback
            this.onUpdate(data);
            
        } catch (error) {
            console.error('Error updating vision features:', error);
            this.clearOverlay();
        }
    }
    
    async toggleUMAP() {
        if (!this.imageId) return;
        
        if (this.isUMAPActive) {
            // Turn off UMAP
            this.isUMAPActive = false;
            this.umapRGBData = null;
            await this.updateVisualization();
            return false;
        } else {
            // Turn on UMAP
            try {
                const response = await fetch(`/api/features/${this.imageId}/umap-rgb`);
                
                if (!response.ok) {
                    throw new Error(`Failed to compute UMAP: ${response.status}`);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Store RGB data
                this.umapRGBData = data.rgb_values;
                this.isUMAPActive = true;
                
                // Apply UMAP visualization
                this.applyUMAPVisualization();
                
                return true;
                
            } catch (error) {
                console.error('Error loading UMAP:', error);
                return false;
            }
        }
    }
    
    applyUMAPVisualization() {
        if (!this.umapRGBData) return;
        
        // Use a higher resolution for better quality
        const size = 480;  // 24x24 -> 480x480
        const scale = size / 24;
        
        // Create canvas to apply alpha
        const canvas = document.createElement('canvas');
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext('2d');
        
        // Create ImageData
        const imageData = ctx.createImageData(size, size);
        const data = imageData.data;
        
        // Fill with RGB values and current alpha
        for (let y = 0; y < size; y++) {
            for (let x = 0; x < size; x++) {
                const srcY = Math.floor(y / scale);
                const srcX = Math.floor(x / scale);
                const srcIdx = srcY * 24 + srcX;
                
                const dstIdx = (y * size + x) * 4;
                data[dstIdx] = this.umapRGBData[srcIdx * 3];      // R
                data[dstIdx + 1] = this.umapRGBData[srcIdx * 3 + 1];  // G
                data[dstIdx + 2] = this.umapRGBData[srcIdx * 3 + 2];  // B
                data[dstIdx + 3] = Math.floor(this.alpha * 255);  // A
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        // Convert to base64 and display
        if (this.overlayElement) {
            this.overlayElement.src = canvas.toDataURL('image/png');
            
            // Apply aspect ratio stretching
            if (this.aspectRatio > 1) {
                this.overlayElement.style.transform = `scaleX(${this.aspectRatio})`;
            } else if (this.aspectRatio < 1) {
                this.overlayElement.style.transform = `scaleY(${1 / this.aspectRatio})`;
            } else {
                this.overlayElement.style.transform = 'none';
            }
        }
        
        if (this.overlayContainer) {
            this.overlayContainer.style.display = 'block';
            this.overlayContainer.style.opacity = '1';
        }
    }
    
    updateStats(stats) {
        if (!this.statsContainer) return;
        
        if (!stats) {
            this.statsContainer.innerHTML = `
                <div class="stat-item">
                    <span class="stat-label">Vision Features:</span>
                    <span class="stat-value">Not Available</span>
                </div>
            `;
            return;
        }
        
        this.statsContainer.innerHTML = `
            <div class="stat-item">
                <span class="stat-label">Max Attention:</span>
                <span class="stat-value">${stats.max_attention || 'N/A'}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Mean Attention:</span>
                <span class="stat-value">${stats.mean_attention || 'N/A'}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Spatial Diversity:</span>
                <span class="stat-value">${stats.spatial_diversity || 'N/A'}</span>
            </div>
            <div class="stat-item">
                <span class="stat-label">Temporal Stability:</span>
                <span class="stat-value">${stats.temporal_stability || 'N/A'}</span>
            </div>
        `;
    }
    
    // Settings update methods
    setTemporalMode(mode) {
        this.temporalMode = mode;
        if (!this.isUMAPActive) {
            this.updateVisualization();
        }
    }
    
    setTemporalFrame(frame) {
        this.temporalFrame = frame;
        if (!this.isUMAPActive && this.temporalMode === 'temporal') {
            this.updateVisualization();
        }
    }
    
    setVisualization(method) {
        this.visualization = method;
        if (!this.isUMAPActive) {
            this.updateVisualization();
        }
    }
    
    setColormap(colormap) {
        this.colormap = colormap;
        if (!this.isUMAPActive) {
            this.updateVisualization();
        }
    }
    
    setAlpha(alpha) {
        this.alpha = alpha;
        if (this.isUMAPActive) {
            this.applyUMAPVisualization();
        } else {
            this.updateVisualization();
        }
    }
    
    toggleOverlay(show) {
        if (this.overlayContainer) {
            this.overlayContainer.style.display = show ? 'block' : 'none';
        }
    }
}

// Export for use in main dashboard
window.VisionFeatureManager = VisionFeatureManager;