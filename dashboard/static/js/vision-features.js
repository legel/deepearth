// Unified Vision Feature Manager for DeepEarth Dashboard
// This module provides a consistent vision feature interface for both Geospatial and Embeddings views

// Colormap definitions for client-side rendering
const colormaps = {
    plasma: [
        [0.050383, 0.029803, 0.527975],
        [0.127568, 0.016298, 0.531895],
        [0.201225, 0.018006, 0.526563],
        [0.269783, 0.038571, 0.509394],
        [0.332553, 0.068007, 0.481904],
        [0.390164, 0.100235, 0.448018],
        [0.443983, 0.133743, 0.410665],
        [0.494897, 0.168256, 0.372237],
        [0.543552, 0.203484, 0.334238],
        [0.590404, 0.239464, 0.297772],
        [0.635682, 0.276349, 0.263448],
        [0.679483, 0.314346, 0.231674],
        [0.721817, 0.353686, 0.202595],
        [0.762651, 0.394583, 0.176184],
        [0.801918, 0.437221, 0.152278],
        [0.839510, 0.481759, 0.130609],
        [0.875302, 0.528309, 0.110859],
        [0.909146, 0.576936, 0.092590],
        [0.940875, 0.627628, 0.074176],
        [0.972752, 0.732803, 0.115965]
    ],
    viridis: [
        [0.267004, 0.004874, 0.329415],
        [0.282623, 0.100196, 0.380271],
        [0.287148, 0.162227, 0.418601],
        [0.284081, 0.212038, 0.449619],
        [0.275305, 0.254901, 0.474538],
        [0.262435, 0.292613, 0.494324],
        [0.247092, 0.326067, 0.509503],
        [0.230299, 0.356022, 0.520561],
        [0.212527, 0.383060, 0.527975],
        [0.194312, 0.407685, 0.532263],
        [0.176381, 0.430382, 0.533967],
        [0.159337, 0.451616, 0.533613],
        [0.144061, 0.471840, 0.531693],
        [0.132449, 0.491514, 0.528655],
        [0.127568, 0.511113, 0.524821],
        [0.134692, 0.531104, 0.520486],
        [0.159337, 0.551956, 0.515896],
        [0.208329, 0.574149, 0.511206],
        [0.287148, 0.598216, 0.506485],
        [0.404070, 0.624694, 0.501712]
    ],
    RdBu_r: [
        [0.403922, 0.000000, 0.121569],
        [0.698039, 0.094118, 0.168627],
        [0.839216, 0.243137, 0.188235],
        [0.925490, 0.407843, 0.321569],
        [0.972549, 0.572549, 0.470588],
        [0.988235, 0.733333, 0.631373],
        [0.988235, 0.858824, 0.780392],
        [0.968627, 0.952941, 0.952941],
        [0.850980, 0.913725, 0.945098],
        [0.698039, 0.839216, 0.921569],
        [0.541176, 0.760784, 0.890196],
        [0.407843, 0.674510, 0.854902],
        [0.262745, 0.576471, 0.764706],
        [0.188235, 0.478431, 0.678431],
        [0.129412, 0.400000, 0.674510],
        [0.019608, 0.188235, 0.380392]
    ]
};

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
        
        // Cache for PCA data
        this.pcaDataCache = null;
        this.cachedImageId = null;
        
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
        
        // Clear cache when loading new image
        this.pcaDataCache = null;
        this.cachedImageId = null;
        
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
            // For PCA visualizations, use the fast endpoint
            if (this.visualization.startsWith('pca')) {
                await this.updatePCAVisualization();
            } else {
                // For other visualizations, use the original endpoint
                await this.updateOtherVisualization();
            }
        } catch (error) {
            console.error('Error updating vision features:', error);
            this.clearOverlay();
        }
    }
    
    async updatePCAVisualization() {
        // Use cached data if available for the same image
        if (this.cachedImageId === this.imageId && this.pcaDataCache) {
            console.log('Using cached PCA data for instant update');
            await this.renderPCAOverlay(this.pcaDataCache);
            return;
        }
        
        // Fetch raw PCA data from fast endpoint
        console.log(`Fetching PCA data for ${this.imageId}`);
        const response = await fetch(`/api/features/${this.imageId}/pca-raw`);
        
        if (!response.ok) {
            console.warn(`No PCA features available for ${this.imageId}`);
            this.clearOverlay();
            this.updateStats(null);
            return;
        }
        
        const data = await response.json();
        
        // Cache the data
        this.pcaDataCache = data;
        this.cachedImageId = this.imageId;
        
        // Render the overlay
        await this.renderPCAOverlay(data);
    }
    
    async renderPCAOverlay(data) {
        const pcaValues = data.pca_values;
        if (!pcaValues || pcaValues.length === 0) return;
        
        // Extract the requested PCA component
        const componentIndex = parseInt(this.visualization.replace('pca', '')) - 1;
        
        // Create canvas for rendering
        const canvas = document.createElement('canvas');
        canvas.width = 384;  // Match server-side dimensions
        canvas.height = 384;
        const ctx = canvas.getContext('2d');
        
        // Get colormap
        const cmap = colormaps[this.colormap] || colormaps.plasma;
        
        // Find min/max for normalization
        let minVal = Infinity, maxVal = -Infinity;
        for (let i = 0; i < 24; i++) {
            for (let j = 0; j < 24; j++) {
                const val = pcaValues[i][j][componentIndex];
                minVal = Math.min(minVal, val);
                maxVal = Math.max(maxVal, val);
            }
        }
        
        // Render the heatmap
        const imageData = ctx.createImageData(384, 384);
        const data_array = imageData.data;
        
        for (let y = 0; y < 384; y++) {
            for (let x = 0; x < 384; x++) {
                const srcY = Math.floor(y * 24 / 384);
                const srcX = Math.floor(x * 24 / 384);
                const value = pcaValues[srcY][srcX][componentIndex];
                
                // Normalize to 0-1
                const normalized = (value - minVal) / (maxVal - minVal);
                
                // Get color from colormap
                const cmapIndex = Math.floor(normalized * (cmap.length - 1));
                const color = cmap[cmapIndex];
                
                const idx = (y * 384 + x) * 4;
                data_array[idx] = Math.floor(color[0] * 255);
                data_array[idx + 1] = Math.floor(color[1] * 255);
                data_array[idx + 2] = Math.floor(color[2] * 255);
                data_array[idx + 3] = Math.floor(this.alpha * 255);
            }
        }
        
        ctx.putImageData(imageData, 0, 0);
        
        // Update overlay
        this.overlayElement.src = canvas.toDataURL('image/png');
        
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
        
        // Update stats if available
        if (data.timing) {
            this.updateStats({
                computation_time: `${data.timing.total_ms}ms`,
                pca_time: `${data.timing.pca_ms}ms`
            });
        }
        
        // Callback
        this.onUpdate(data);
    }
    
    async updateOtherVisualization() {
        // For non-PCA visualizations, use the original endpoint
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
        const wasPCA = this.visualization.startsWith('pca');
        this.visualization = method;
        if (!this.isUMAPActive) {
            // If switching between PCA components and we have cached data, just re-render
            if (method.startsWith('pca') && this.pcaDataCache && wasPCA) {
                console.log('Switching PCA component with cached data');
                this.renderPCAOverlay(this.pcaDataCache);
            } else {
                // Clear cache if switching away from PCA
                if (!method.startsWith('pca')) {
                    this.pcaDataCache = null;
                    this.cachedImageId = null;
                }
                this.updateVisualization();
            }
        }
    }
    
    setColormap(colormap) {
        this.colormap = colormap;
        if (!this.isUMAPActive) {
            // For PCA with cached data, just re-render
            if (this.visualization.startsWith('pca') && this.pcaDataCache) {
                console.log('Re-rendering PCA with new colormap (no fetch needed)');
                this.renderPCAOverlay(this.pcaDataCache);
            } else {
                this.updateVisualization();
            }
        }
    }
    
    setAlpha(alpha) {
        this.alpha = alpha;
        if (this.isUMAPActive) {
            this.applyUMAPVisualization();
        } else {
            // For PCA with cached data, just re-render
            if (this.visualization.startsWith('pca') && this.pcaDataCache) {
                console.log('Re-rendering PCA with new alpha (no fetch needed)');
                this.renderPCAOverlay(this.pcaDataCache);
            } else {
                this.updateVisualization();
            }
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