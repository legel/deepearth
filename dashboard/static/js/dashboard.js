// DeepEarth Unified Dashboard JavaScript

// Global variables
let map;
let observationsData = [];
let markersLayer;
let gridLayer;
let selectedGrid = null;
let scene3D, camera3D, renderer3D;
let embeddingPoints = [];
let currentView = 'geospatial';
let currentObservation = null;
let currentImages = [];
let currentImageIndex = 0;
let yearlyChart = null;
let speciesUmapColors = {}; // Store UMAP-based colors for species
let selectedMarker = null; // Track selected marker

// 3D View variables
let currentEmbeddingView = 'language';
let geographicBounds = null;
let visionEmbeddings = [];
let languageEmbeddings = [];
let animationParams = {
    speed: 2000,
    loopiness: 0.5,
    stagger: 20
};
let debugMode = false;
let currentPointData = null;
let galleryImages = [];
let galleryObservations = []; // Store observation list for on-demand loading
let galleryIndex = 0;

// Vision embeddings preloading
let availableVisionEmbeddings = null;
let isPreloadingVision = false;
let loadGeneration = 0; // Track load operations to prevent race conditions

// Vision feature variables
let currentTemporalMode = 'mean';
let temporalFrames = [];
let currentColormap = 'plasma';
let currentAlpha = 0.7;
let currentVisualization = 'pca1';
let autoShowUMAP = false;
let isUMAPActive = false;
let umapRGBData = null;
let currentObservationId = null;
let currentImageAspectRatio = 1;
let currentPCAData = null; // Store PCA data to avoid refetching

// Performance monitoring for debugging
const performanceMonitor = {
    timers: {},
    start: function(label) {
        this.timers[label] = performance.now();
        console.log(`⏱️ START: ${label} at ${new Date().toISOString()}`);
    },
    end: function(label) {
        if (this.timers[label]) {
            const duration = performance.now() - this.timers[label];
            console.log(`⏱️ END: ${label} took ${duration.toFixed(2)}ms`);
            delete this.timers[label];
            return duration;
        }
        return 0;
    },
    log: function(message, data = {}) {
        console.log(`🐛 DEBUG: ${message}`, {
            ...data,
            timestamp: new Date().toISOString(),
            url: window.location.href,
            userAgent: navigator.userAgent.substring(0, 50) + '...'
        });
    }
};

// Colormaps are now defined in vision-features.js as window.colormaps

// Client-side PCA overlay generation
async function generatePCAOverlay(pcaGrid, colormap, alpha) {
    const size = 384; // Display size
    const scale = size / 24; // Scale from 24x24 to display size
    
    // Create canvas
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d');
    
    // Get colormap - handle case variations
    const cmapName = colormap === 'RdBu_r' ? 'RdBu_r' : colormap.toLowerCase();
    const cmap = window.colormaps[cmapName] || window.colormaps.plasma;
    
    // Create ImageData
    const imageData = ctx.createImageData(size, size);
    const data = imageData.data;
    
    // Fill with PCA values using nearest neighbor upsampling
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const srcY = Math.floor(y / scale);
            const srcX = Math.floor(x / scale);
            
            // Get PCA value
            const value = pcaGrid[srcY][srcX];
            
            // Map to colormap
            const cmapIdx = Math.floor(value * (cmap.length - 1));
            const color = cmap[Math.min(cmapIdx, cmap.length - 1)];
            
            // Set pixel
            const idx = (y * size + x) * 4;
            data[idx] = Math.floor(color[0] * 255);
            data[idx + 1] = Math.floor(color[1] * 255);
            data[idx + 2] = Math.floor(color[2] * 255);
            data[idx + 3] = Math.floor(alpha * 255);
        }
    }
    
    // Put image data
    ctx.putImageData(imageData, 0, 0);
    
    // Convert to data URL
    return canvas.toDataURL('image/png');
}

// Initialize vision feature UI defaults
function initializeVisionFeatureDefaults() {
    console.log('🎨 Initializing vision feature UI defaults (PCA1 + Plasma)');
    
    // Set PCA1 button as active
    const pca1Btn = document.getElementById('pca1-btn');
    if (pca1Btn) {
        // Remove active from all visualization buttons first
        document.querySelectorAll('.visualization-btn').forEach(btn => btn.classList.remove('active'));
        pca1Btn.classList.add('active');
        console.log('✅ PCA1 button set as active');
    }
    
    // Set Plasma button as active
    const plasmaBtn = document.getElementById('plasma-btn');
    if (plasmaBtn) {
        // Remove active from all colormap buttons first
        document.querySelectorAll('.colormap-btn').forEach(btn => btn.classList.remove('active'));
        plasmaBtn.classList.add('active');
        console.log('✅ Plasma button set as active');
    }
    
    // Set mean temporal mode as active
    const meanBtn = document.getElementById('mean-btn');
    if (meanBtn) {
        document.querySelectorAll('.temporal-btn').forEach(btn => btn.classList.remove('active'));
        meanBtn.classList.add('active');
        console.log('✅ Mean temporal mode set as active');
    }
    
    // Initialize alpha slider display
    const alphaValue = document.getElementById('alphaValue');
    if (alphaValue) {
        alphaValue.textContent = `${Math.round(currentAlpha * 100)}%`;
    }
    
    // Initialize alpha slider value
    const alphaSlider = document.getElementById('alpha-slider');
    if (alphaSlider) {
        alphaSlider.value = Math.round(currentAlpha * 100);
    }
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    // Force console to be visible and working
    console.clear();
    console.log('🚀 DASHBOARD STARTING - DeepEarth v2.0');
    console.log('🌍 Initializing DeepEarth Dashboard...');
    
    // Test console logging
    console.warn('⚡ Console logging is working');
    console.error('🔥 Error logging test (this is not a real error)');
    
    // Log browser info for debugging
    console.log('🔧 Browser info:', {
        userAgent: navigator.userAgent,
        viewport: { width: window.innerWidth, height: window.innerHeight },
        timestamp: new Date().toISOString()
    });
    
    // Force initial embedding view to language
    currentEmbeddingView = 'language';
    console.log('Set initial currentEmbeddingView to language');
    
    // Initialize filter state
    if (window.filterState) {
        // Apply saved filters to UI
        window.filterState.applyToUI();
        
        // Listen for filter changes
        window.filterState.addListener((type, data, allFilters) => {
            console.log('Filter changed:', type, data);
            
            // Clear preloaded data when filters change
            availableVisionEmbeddings = null;
            
            // Don't clear the unfiltered cache, as it's still valid
            // Only clear if we're resetting to defaults
            if (type === 'reset') {
                // Keep the unfiltered cache even on reset
            }
            
            // Update relevant views
            if (currentView === 'geospatial') {
                filterObservations();
                updateStatistics();
            } else if (currentView === 'ecological') {
                if (currentEmbeddingView === 'vision') {
                    // Preload then compute
                    preloadAvailableVisionEmbeddings().then(() => {
                        computeVisionUMAP();
                    });
                } else {
                    loadLanguageEmbeddings();
                }
            }
        });
    }
    
    // Initialize view switcher
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            switchView(this.dataset.view);
        });
    });
    
    // Initialize controls
    initializeControls();
    
    // Initialize 3D view controls
    initialize3DControls();
    
    // Load initial data
    loadObservations();
    
    // Initialize map
    initializeMap();
    
    // Initialize panel resize functionality
    initializePanelResize();
    
    // Ensure we start with language embeddings
    currentEmbeddingView = 'language';
    console.log('Confirmed initial embedding view is language');
    
    // Initialize vision feature defaults (PCA1 and Plasma as requested)
    initializeVisionFeatureDefaults();
    
    // Initialize gallery vision manager
    if (window.VisionFeatureManager) {
        galleryVisionManager = new VisionFeatureManager({
            container: document.getElementById('point-image-gallery'),
            imageElement: document.getElementById('gallery-image'),
            overlayElement: document.getElementById('gallery-attention-img'),
            overlayContainer: document.getElementById('gallery-attention-overlay'),
            statsContainer: null, // Will be set dynamically when stats are available
            onUpdate: (data) => {
                console.log('Gallery vision features updated', data);
                // Update gallery stats if available
                if (data && data.stats) {
                    updateGalleryStats(data.stats);
                }
            }
        });
    }
    
    // Enable debug mode with URL parameter
    if (window.location.search.includes('debug=true')) {
        enableDebugMode();
    }
    
    // Add keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + R to force recompute UMAP
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && currentView === 'ecological') {
            e.preventDefault();
            console.log('Force recomputing UMAP...');
            if (currentEmbeddingView === 'language') {
                loadLanguageEmbeddings(window.lastSelectedSpecies, true);
            }
        }
    });
});

// Sync temporal filters between views
function syncTemporalFilters(fromEmbeddings = false) {
    if (fromEmbeddings) {
        // Copy from embeddings to geospatial
        document.getElementById('year-min').value = document.getElementById('emb-year-min').value;
        document.getElementById('year-max').value = document.getElementById('emb-year-max').value;
        document.getElementById('month-min').value = document.getElementById('emb-month-min').value;
        document.getElementById('month-max').value = document.getElementById('emb-month-max').value;
        document.getElementById('hour-min').value = document.getElementById('emb-hour-min').value;
        document.getElementById('hour-max').value = document.getElementById('emb-hour-max').value;
    } else {
        // Copy from geospatial to embeddings
        document.getElementById('emb-year-min').value = document.getElementById('year-min').value;
        document.getElementById('emb-year-max').value = document.getElementById('year-max').value;
        document.getElementById('emb-month-min').value = document.getElementById('month-min').value;
        document.getElementById('emb-month-max').value = document.getElementById('month-max').value;
        document.getElementById('emb-hour-min').value = document.getElementById('hour-min').value;
        document.getElementById('emb-hour-max').value = document.getElementById('hour-max').value;
    }
}

// View switching
function switchView(view) {
    currentView = view;
    
    console.log('Switching view to:', view);
    
    // Update buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.view === view);
    });
    
    // Update views
    document.getElementById('geospatial-view').classList.toggle('active', view === 'geospatial');
    document.getElementById('ecological-view').classList.toggle('active', view === 'ecological');
    
    // Update controls
    document.getElementById('geospatial-controls').style.display = view === 'geospatial' ? 'block' : 'none';
    document.getElementById('ecological-controls').style.display = view === 'ecological' ? 'block' : 'none';
    
    if (view === 'ecological') {
        // Close observation panel when switching to embeddings view
        const observationPanel = document.getElementById('observation-panel');
        if (observationPanel) {
            observationPanel.style.display = 'none';
        }
        
        if (!scene3D) {
            initialize3DView();
        }
        
        // Apply current filters to UI
        if (window.filterState) {
            window.filterState.applyToUI();
        }
        
        // Force currentEmbeddingView to language before selecting
        currentEmbeddingView = 'language';
        console.log('Forcing embedding view to language');
        
        // Cancel any pending vision operations
        loadGeneration++;
        
        // Always default to language embeddings when switching to embeddings view
        selectEmbeddingType('language');
        
        // Don't preload vision embeddings when switching to embeddings view
        // User should explicitly click on vision embeddings button
        console.log('Switched to embeddings view, showing language embeddings');
        
        // Mark that we're now in embeddings view
        window.lastViewWasGeospatial = false;
    } else if (view === 'geospatial') {
        // Mark that we're in geospatial view
        window.lastViewWasGeospatial = true;
        
        // Update statistics for geospatial view
        updateStatistics();
        
        // Apply current filters to UI
        if (window.filterState) {
            window.filterState.applyToUI();
        }
        
        // Fix map display issues
        if (map) {
            setTimeout(() => {
                map.invalidateSize();
                
                // Check if we were looking at a specific observation in embeddings
                if (window.lastSelectedGbifId) {
                    // Find the observation marker and select it
                    const obs = observationsData.find(o => o.gbif_id === window.lastSelectedGbifId);
                    if (obs) {
                        // Fly to the observation
                        if (obs.decimalLatitude && obs.decimalLongitude) {
                            map.flyTo([obs.decimalLatitude, obs.decimalLongitude], 14, {
                                duration: 1.5
                            });
                        }
                        
                        // Simulate click on the marker after a short delay
                        setTimeout(() => {
                            if (markersLayer) {
                                markersLayer.eachLayer(marker => {
                                    if (marker.observation && marker.observation.gbif_id === window.lastSelectedGbifId) {
                                        marker.fire('click');
                                    }
                                });
                            }
                        }, 1600);
                    }
                }
            }, 100);
        }
    }
}

// Initialize 3D view controls
function initialize3DControls() {
    // Max images slider
    const maxImagesSlider = document.getElementById('max-images-slider');
    const maxImagesValue = document.getElementById('max-images-value');
    if (maxImagesSlider) {
        maxImagesSlider.addEventListener('input', function(e) {
            maxImagesValue.textContent = e.target.value;
            
            // Clear preloaded data when max images changes
            availableVisionEmbeddings = null;
            
            // Preload new data if in vision mode
            if (currentView === 'ecological' && currentEmbeddingView === 'vision') {
                // Debounce the preloading
                clearTimeout(window.preloadDebounce);
                window.preloadDebounce = setTimeout(() => {
                    preloadAvailableVisionEmbeddings();
                }, 500);
            }
        });
    }
    
    // Clear filter button
    const clearFilterBtn = document.getElementById('clear-filter');
    if (clearFilterBtn) {
        clearFilterBtn.addEventListener('click', clearGeographicFilter);
    }
    
    // Debug mode sliders
    const animSpeedSlider = document.getElementById('animation-speed');
    if (animSpeedSlider) {
        animSpeedSlider.addEventListener('input', function(e) {
            animationParams.speed = parseInt(e.target.value);
            document.getElementById('speed-value').textContent = e.target.value + 'ms';
            if (debugMode) {
                console.log('Animation speed:', animationParams.speed);
            }
        });
    }
    
    const loopinessSlider = document.getElementById('curve-loopiness');
    if (loopinessSlider) {
        loopinessSlider.addEventListener('input', function(e) {
            animationParams.loopiness = parseFloat(e.target.value);
            document.getElementById('loopiness-value').textContent = e.target.value;
            if (debugMode) {
                console.log('Curve loopiness:', animationParams.loopiness);
            }
        });
    }
    
    const staggerSlider = document.getElementById('stagger-delay');
    if (staggerSlider) {
        staggerSlider.addEventListener('input', function(e) {
            animationParams.stagger = parseInt(e.target.value);
            document.getElementById('stagger-value').textContent = e.target.value + 'ms';
            if (debugMode) {
                console.log('Stagger delay:', animationParams.stagger);
            }
        });
    }
    
    // Keyboard shortcuts for gallery
    document.addEventListener('keydown', function(e) {
        if (document.getElementById('point-info-panel').style.display === 'block') {
            if (e.key === 'ArrowLeft') {
                navigateGallery(-1);
            } else if (e.key === 'ArrowRight') {
                navigateGallery(1);
            } else if (e.key === 'Escape') {
                closePointPanel();
            }
        }
    });
}

// Initialize controls
function initializeControls() {
    // Set default visualization methods
    setTimeout(() => {
        if (document.getElementById('visualizationMethod')) {
            document.getElementById('visualizationMethod').value = 'pca1';
            currentVisualization = 'pca1';
        }
        if (document.getElementById('galleryVisualizationMethod')) {
            document.getElementById('galleryVisualizationMethod').value = 'pca1';
            galleryCurrentVisualization = 'pca1';
        }
    }, 100);
    
    // Grid toggle
    document.getElementById('show-grid').addEventListener('change', function(e) {
        document.querySelector('.grid-controls').style.display = e.target.checked ? 'block' : 'none';
        if (e.target.checked) {
            drawGrid();
        } else {
            clearGrid();
            // Close grid stats overlay when grid is turned off
            const gridStatsOverlay = document.getElementById('grid-stats-overlay');
            if (gridStatsOverlay) {
                gridStatsOverlay.style.display = 'none';
            }
        }
    });
    
    // Grid size slider
    document.getElementById('grid-size').addEventListener('input', function(e) {
        document.getElementById('grid-size-value').textContent = e.target.value + ' km';
        if (document.getElementById('show-grid').checked) {
            drawGrid();
        }
    });
    
    // Connect temporal filters to filter state
    const temporalInputs = [
        { id: 'year-min', type: 'yearMin' },
        { id: 'year-max', type: 'yearMax' },
        { id: 'month-min', type: 'monthMin' },
        { id: 'month-max', type: 'monthMax' },
        { id: 'hour-min', type: 'hourMin' },
        { id: 'hour-max', type: 'hourMax' },
        { id: 'emb-year-min', type: 'yearMin' },
        { id: 'emb-year-max', type: 'yearMax' },
        { id: 'emb-month-min', type: 'monthMin' },
        { id: 'emb-month-max', type: 'monthMax' },
        { id: 'emb-hour-min', type: 'hourMin' },
        { id: 'emb-hour-max', type: 'hourMax' }
    ];
    
    temporalInputs.forEach(({ id, type }) => {
        const input = document.getElementById(id);
        if (input) {
            input.addEventListener('change', function() {
                if (window.filterState) {
                    window.filterState.setTemporalFilter(type, parseInt(this.value));
                }
            });
        }
    });
    
    // Date/time range filters
    ['year-min', 'year-max', 'month-min', 'month-max', 'hour-min', 'hour-max'].forEach(id => {
        const elem = document.getElementById(id);
        if (elem) {
            elem.addEventListener('change', () => {
                syncTemporalFilters();
                filterObservations();
            });
        }
    });
    
    // Embeddings view temporal filters
    ['emb-year-min', 'emb-year-max', 'emb-month-min', 'emb-month-max', 'emb-hour-min', 'emb-hour-max'].forEach(id => {
        const elem = document.getElementById(id);
        if (elem) {
            elem.addEventListener('change', () => {
                syncTemporalFilters(true);
                // Recompute embeddings with new filters
                if (currentView === 'ecological') {
                    if (currentEmbeddingView === 'language') {
                        loadLanguageEmbeddings(window.lastSelectedSpecies);
                    } else {
                        computeVisionUMAP();
                    }
                }
            });
        }
    });
    
    // Vision only filter
    document.getElementById('show-vision-only').addEventListener('change', filterObservations);
    
    // Base layer selector
    document.getElementById('base-layer-select').addEventListener('change', function(e) {
        changeBaseLayer(e.target.value);
    });
}

// Update map bounds display
function updateBoundsDisplay() {
    if (!map) return;
    
    const bounds = map.getBounds();
    const north = bounds.getNorth();
    const south = bounds.getSouth();
    const east = bounds.getEast();
    const west = bounds.getWest();
    
    // Update the display
    const latBounds = document.getElementById('lat-bounds');
    const lonBounds = document.getElementById('lon-bounds');
    
    if (latBounds) {
        latBounds.textContent = `${south.toFixed(4)}, ${north.toFixed(4)}`;
    }
    
    if (lonBounds) {
        lonBounds.textContent = `${west.toFixed(4)}, ${east.toFixed(4)}`;
    }
}

// Initialize map
function initializeMap() {
    console.log('Initializing map...');
    
    // Create map centered on Central Florida
    map = L.map('map').setView([28.5, -81.4], 9);
    
    // Add satellite base layer
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri',
        maxZoom: 19
    }).addTo(map);
    
    // Create markers layer group
    markersLayer = L.layerGroup().addTo(map);
    
    // Update bounds display on map move
    map.on('moveend', updateBoundsDisplay);
    map.on('zoomend', updateBoundsDisplay);
    
    // Initial bounds update
    updateBoundsDisplay();
}

// Initialize panel resize functionality
function initializePanelResize() {
    const panel = document.getElementById('observation-panel');
    const handle = document.getElementById('panel-resize-handle');
    const container = document.getElementById('geospatial-view');
    
    if (!panel || !handle) return;
    
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
    
    // Get stored panel width or use default
    const storedWidth = localStorage.getItem('panelWidth');
    if (storedWidth) {
        document.documentElement.style.setProperty('--panel-width', storedWidth);
    }
    
    handle.addEventListener('mousedown', (e) => {
        isResizing = true;
        startX = e.clientX;
        startWidth = panel.offsetWidth;
        handle.classList.add('dragging');
        document.body.style.cursor = 'ew-resize';
        document.body.style.userSelect = 'none';
        e.preventDefault();
    });
    
    document.addEventListener('mousemove', (e) => {
        if (!isResizing) return;
        
        const diff = startX - e.clientX;
        const newWidth = startWidth + diff;
        const containerWidth = container.offsetWidth;
        const widthPercent = (newWidth / containerWidth) * 100;
        
        // Constrain between min and max
        const minPercent = (300 / containerWidth) * 100;
        const maxPercent = 60;
        
        if (widthPercent >= minPercent && widthPercent <= maxPercent) {
            document.documentElement.style.setProperty('--panel-width', widthPercent + '%');
            
            // Update map size in real-time
            if (map) {
                map.invalidateSize();
            }
        }
    });
    
    document.addEventListener('mouseup', () => {
        if (isResizing) {
            isResizing = false;
            handle.classList.remove('dragging');
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            
            // Save panel width to localStorage
            const panelWidth = getComputedStyle(document.documentElement).getPropertyValue('--panel-width');
            if (panelWidth) {
                localStorage.setItem('panelWidth', panelWidth.trim());
            }
        }
    });
}

// Load observations data
async function loadObservations() {
    try {
        showLoading(true);
        
        // Load observations and species colors in parallel
        const [obsResponse, colorsResponse] = await Promise.all([
            fetch('/api/observations'),
            fetch('/api/species_umap_colors')
        ]);
        
        const data = await obsResponse.json();
        const colorsData = await colorsResponse.json();
        
        observationsData = data.observations;
        speciesUmapColors = colorsData.taxon_colors || {};
        
        console.log(`Loaded ${observationsData.length} observations`);
        console.log(`Loaded colors for ${Object.keys(speciesUmapColors).length} species`);
        
        // Update statistics
        updateStatistics();
        
        // Display observations on map
        displayObservations();
        
        // Fit map to bounds
        if (data.bounds) {
            map.fitBounds([
                [data.bounds.south, data.bounds.west],
                [data.bounds.north, data.bounds.east]
            ]);
        }
        
        showLoading(false);
    } catch (error) {
        console.error('Error loading observations:', error);
        showLoading(false);
    }
}

// Update statistics
function updateStatistics() {
    // Get filtered observations based on current filters
    const filtered = getFilteredObservations();
    const uniqueSpecies = new Set(filtered.map(o => o.taxon_id)).size;
    const withImages = filtered.filter(o => o.has_vision).length;
    
    document.getElementById('total-observations').textContent = filtered.length.toLocaleString();
    document.getElementById('total-species').textContent = uniqueSpecies.toLocaleString();
    document.getElementById('total-images').textContent = withImages.toLocaleString();
}

// Display observations on map
function displayObservations() {
    markersLayer.clearLayers();
    
    const filtered = getFilteredObservations();
    console.log(`Displaying ${filtered.length} filtered observations`);
    
    // Create markers for each observation
    filtered.forEach(obs => {
        // Use UMAP RGB color if available, otherwise default based on vision
        let color = '#94a3b8'; // Default gray
        if (speciesUmapColors[obs.taxon_id]) {
            color = speciesUmapColors[obs.taxon_id].hex;
        } else if (obs.has_vision) {
            color = '#22c55e'; // Green for observations with images
        }
        
        const marker = L.circleMarker([obs.lat, obs.lon], {
            radius: 6,
            fillColor: color,
            color: '#ffffff',
            weight: 1,
            opacity: 1,
            fillOpacity: 0.8
        });
        
        // Store observation data on marker
        marker.observationData = obs;
        
        marker.bindPopup(createPopupContent(obs));
        marker.on('click', () => {
            showObservationDetails(obs, marker);
            // Update popup after GBIF data loads
            setTimeout(() => {
                if (marker.isPopupOpen()) {
                    marker.setPopupContent(createPopupContent(obs));
                }
            }, 1000);
        });
        
        markersLayer.addLayer(marker);
    });
}

// Filter observations based on controls
function getFilteredObservations() {
    let filtered = observationsData;
    
    // Year range filter
    const yearMin = parseInt(document.getElementById('year-min').value);
    const yearMax = parseInt(document.getElementById('year-max').value);
    filtered = filtered.filter(o => o.year >= yearMin && o.year <= yearMax);
    
    // Month range filter
    const monthMin = parseInt(document.getElementById('month-min').value);
    const monthMax = parseInt(document.getElementById('month-max').value);
    filtered = filtered.filter(o => !o.month || (o.month >= monthMin && o.month <= monthMax));
    
    // Hour range filter
    const hourMin = parseInt(document.getElementById('hour-min').value);
    const hourMax = parseInt(document.getElementById('hour-max').value);
    filtered = filtered.filter(o => o.hour === null || o.hour === undefined || (o.hour >= hourMin && o.hour <= hourMax));
    
    // Vision only filter
    if (document.getElementById('show-vision-only').checked) {
        filtered = filtered.filter(o => o.has_vision);
    }
    
    return filtered;
}

// Filter and redisplay
function filterObservations() {
    displayObservations();
    updateStatistics();
}

// Create popup content
function createPopupContent(obs) {
    const date = formatDate(obs);
    const time = formatTime(obs);
    
    // Check if we have GBIF data
    let gbifInfo = '';
    if (currentObservation && currentObservation.gbifData && currentObservation.gbif_id === obs.gbif_id) {
        const gbif = currentObservation.gbifData;
        const recordedBy = gbif.recordedBy || 'Unknown';
        const license = gbif.license ? gbif.license.replace('http://creativecommons.org/licenses/', 'CC ') : '';
        
        gbifInfo = `
            <div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd;">
                <div>Recorded by: ${recordedBy}</div>
                ${license ? `<div>License: ${license}</div>` : ''}
                <div style="margin-top: 4px;">
                    <a href="https://www.gbif.org/occurrence/${gbif.key}" target="_blank" 
                       style="color: #2563eb; text-decoration: none;">
                        View on GBIF →
                    </a>
                </div>
            </div>
        `;
    }
    
    return `
        <div style="min-width: 200px;">
            <h4 style="margin: 0 0 8px 0;">${obs.taxon_name}</h4>
            <div style="font-size: 12px; color: #666;">
                <div>Date: ${date}</div>
                ${time ? `<div>Time: ${time}</div>` : ''}
                <div>GBIF: ${obs.gbif_id}</div>
                <div style="margin-top: 8px;">
                    ${obs.has_vision ? '✅ Has images' : '❌ No images'}
                </div>
                ${gbifInfo}
            </div>
        </div>
    `;
}

// Format date
function formatDate(obs) {
    if (!obs.year) return 'Unknown';
    let date = obs.year.toString();
    if (obs.month) {
        date += `-${obs.month.toString().padStart(2, '0')}`;
        if (obs.day) {
            date += `-${obs.day.toString().padStart(2, '0')}`;
        }
    }
    return date;
}

// Format time
function formatTime(obs) {
    if (!obs.hour && obs.hour !== 0) return null;
    let time = obs.hour.toString().padStart(2, '0');
    time += ':' + (obs.minute || 0).toString().padStart(2, '0');
    time += ':' + (obs.second || 0).toString().padStart(2, '0');
    return time;
}

// Show observation details
async function showObservationDetails(obs, marker) {
    try {
        // Highlight selected marker
        if (selectedMarker) {
            selectedMarker.setStyle({
                radius: 6,
                weight: 1
            });
        }
        
        if (marker) {
            selectedMarker = marker;
            marker.setStyle({
                radius: 10,
                weight: 3,
                color: '#ffff00'
            });
        }
        
        const response = await fetch(`/api/observation/${obs.gbif_id}`);
        const details = await response.json();
        
        // Log all metadata to console for feature development
        console.log('🔍 GBIF Observation Metadata:', {
            ...obs,  // Original observation data
            ...details,  // Detailed API response
            _timestamp: new Date().toISOString(),
            _source: 'geospatial_click'
        });
        
        currentObservation = details;
        // Store last selected species and GBIF ID
        window.lastSelectedSpecies = details.taxon_name;
        window.lastSelectedGbifId = details.gbif_id;
        
        // Center map on this observation
        if (map && obs.lat && obs.lon) {
            map.setView([obs.lat, obs.lon], Math.max(map.getZoom(), 15), {
                animate: true,
                duration: 0.5
            });
            
            // Update bounds display after animation
            setTimeout(updateBoundsDisplay, 600);
        }
        
        // Query GBIF for additional information
        queryGBIFData(obs.gbif_id);
        
        // Add panel-open class to adjust map width
        document.getElementById('geospatial-view').classList.add('panel-open');
        
        // Invalidate map size after transition
        setTimeout(() => {
            if (map) {
                map.invalidateSize();
            }
        }, 300);
        
        // Update panel content
        document.getElementById('obs-species-name').textContent = details.taxon_name;
        document.getElementById('obs-location').textContent = 
            `${details.location.latitude.toFixed(4)}, ${details.location.longitude.toFixed(4)}`;
        document.getElementById('obs-date').textContent = formatDate(details.temporal);
        document.getElementById('obs-time').textContent = formatTime(details.temporal) || 'Not recorded';
        
        // Show images if available
        if (details.images && details.images.length > 0) {
            currentImages = details.images;
            currentImageIndex = 0;
            
            // Clear any previous overlay before showing new image
            const overlayContainer = document.getElementById('obs-attention-overlay');
            const overlayImg = document.getElementById('obs-attention-img');
            if (overlayImg) {
                overlayImg.src = '';
                overlayImg.style.transform = 'none'; // Reset transform
            }
            if (overlayContainer) overlayContainer.style.display = 'none';
            
            document.getElementById('obs-image').src = details.images[0].url;
            const viewer = document.querySelector('.enhanced-image-viewer');
            if (viewer) {
                viewer.style.display = 'block';
            }
            
            // Load vision features if available
            if (details.images[0].image_id) {
                console.log('🖼️ Loading vision features for image:', {
                    image_id: details.images[0].image_id,
                    gbif_id: details.gbif_id,
                    species: details.taxon_name,
                    has_vision: details.has_vision_embedding
                });
                
                console.log(`🚀 AUTO-LOADING VISION FEATURES for observation:`, {
                    imageId: details.images[0].image_id,
                    gbifId: details.gbif_id,
                    species: details.taxon_name,
                    hasVision: details.has_vision_embedding,
                    timestamp: new Date().toISOString()
                });
                
                loadImageAndFeatures(details.images[0].image_id);
                
                // Auto-show UMAP if it was active before
                if (autoShowUMAP && !isUMAPActive) {
                    console.log(`🌈 Auto-restoring UMAP visualization in 500ms`);
                    setTimeout(() => toggleUMAP(), 500);
                }
            }
        } else {
            const viewer = document.querySelector('.enhanced-image-viewer');
            if (viewer) {
                viewer.style.display = 'none';
            }
        }
        
        // Show panel
        document.getElementById('observation-panel').style.display = 'block';
    } catch (error) {
        console.error('Error loading observation details:', error);
    }
}

// Close observation panel
function closeObservationPanel() {
    document.getElementById('observation-panel').style.display = 'none';
    document.getElementById('geospatial-view').classList.remove('panel-open');
    currentObservation = null;
    
    // Reset highlighted marker
    if (selectedMarker) {
        selectedMarker.setStyle({
            radius: 6,
            weight: 1
        });
        selectedMarker = null;
    }
    
    
    // Invalidate map size after transition
    setTimeout(() => {
        if (map) {
            map.invalidateSize();
        }
    }, 300);
}

// Query GBIF for additional observation data
async function queryGBIFData(gbifId) {
    try {
        // Query GBIF occurrence API
        const response = await fetch(`https://api.gbif.org/v1/occurrence/${gbifId}`);
        if (!response.ok) {
            console.warn('GBIF data not available for this observation');
            return;
        }
        
        const gbifData = await response.json();
        
        // Store GBIF data for popup use
        if (currentObservation) {
            currentObservation.gbifData = gbifData;
        }
        
    } catch (error) {
        console.error('Error querying GBIF:', error);
    }
}


// View features
async function viewFeatures() {
    if (!currentObservation || !currentObservation.has_vision_embedding) return;
    
    try {
        const response = await fetch(`/api/vision_features/${currentObservation.gbif_id}`);
        const features = await response.json();
        
        if (features.attention_map) {
            // Create attention overlay
            const overlay = document.getElementById('attention-overlay');
            overlay.innerHTML = `<img src="${features.attention_map}" style="width: 100%; height: 100%;">`;
        }
    } catch (error) {
        console.error('Error loading vision features:', error);
    }
}

// Grid functionality
function drawGrid() {
    clearGrid();
    
    const bounds = map.getBounds();
    const gridSize = parseInt(document.getElementById('grid-size').value);
    
    // Convert km to degrees (approximate)
    const kmToDeg = gridSize / 111.0;
    
    // Create grid
    gridLayer = L.layerGroup().addTo(map);
    
    const west = Math.floor(bounds.getWest() / kmToDeg) * kmToDeg;
    const east = Math.ceil(bounds.getEast() / kmToDeg) * kmToDeg;
    const south = Math.floor(bounds.getSouth() / kmToDeg) * kmToDeg;
    const north = Math.ceil(bounds.getNorth() / kmToDeg) * kmToDeg;
    
    // Draw grid lines
    for (let lat = south; lat <= north; lat += kmToDeg) {
        const line = L.polyline([[lat, west], [lat, east]], {
            color: 'white',
            weight: 1,
            opacity: 0.5
        }).addTo(gridLayer);
    }
    
    for (let lon = west; lon <= east; lon += kmToDeg) {
        const line = L.polyline([[south, lon], [north, lon]], {
            color: 'white',
            weight: 1,
            opacity: 0.5
        }).addTo(gridLayer);
    }
    
    // Create clickable grid cells
    for (let lat = south; lat < north; lat += kmToDeg) {
        for (let lon = west; lon < east; lon += kmToDeg) {
            const bounds = [[lat, lon], [lat + kmToDeg, lon + kmToDeg]];
            const rect = L.rectangle(bounds, {
                color: 'transparent',
                fillColor: 'transparent',
                fillOpacity: 0
            });
            
            rect.on('click', function(e) {
                selectGridCell(bounds, e);
            });
            
            rect.addTo(gridLayer);
        }
    }
}

function clearGrid() {
    if (gridLayer) {
        map.removeLayer(gridLayer);
        gridLayer = null;
    }
    selectedGrid = null;
}

// Select grid cell
async function selectGridCell(bounds, event) {
    // Clear previous selection
    if (selectedGrid) {
        map.removeLayer(selectedGrid);
    }
    
    // Highlight selected cell
    selectedGrid = L.rectangle(bounds, {
        color: '#2563eb',
        fillColor: '#2563eb',
        fillOpacity: 0.2,
        weight: 2
    }).addTo(map);
    
    // Get statistics for this cell
    const [[south, west], [north, east]] = bounds;
    
    // Update filter state with grid cell bounds
    const gridBounds = { north, south, east, west };
    if (window.filterState) {
        window.filterState.setGridCellFilter(gridBounds);
    }
    
    try {
        // Build URL with filter state parameters
        const params = window.filterState ? window.filterState.getAPIParams() : new URLSearchParams();
        const url = `/api/grid_statistics?${params.toString()}`;
            
        const response = await fetch(url);
        const stats = await response.json();
        
        if (stats.error) {
            console.warn('No data in selected grid cell');
            return;
        }
        
        // Update grid statistics panel
        document.getElementById('grid-species-count').textContent = stats.total_species;
        document.getElementById('grid-obs-count').textContent = stats.total_observations;
        
        // Update yearly chart
        updateYearlyChart(stats.yearly_counts);
        
        // Update species table
        updateSpeciesTable(stats.species);
        
        // Store current bounds for analysis
        window.currentGridBounds = { north, south, east, west };
        
        // Show panel
        document.getElementById('grid-stats-overlay').style.display = 'block';
    } catch (error) {
        console.error('Error getting grid statistics:', error);
    }
}

// Update yearly chart
function updateYearlyChart(yearlyData) {
    const ctx = document.getElementById('yearly-chart').getContext('2d');
    
    if (yearlyChart) {
        yearlyChart.destroy();
    }
    
    yearlyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: yearlyData.map(d => d.year),
            datasets: [{
                label: 'Observations',
                data: yearlyData.map(d => d.count),
                backgroundColor: '#2563eb',
                borderColor: '#1d4ed8',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Observations by Year',
                    color: '#f1f5f9'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: '#475569'
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#94a3b8'
                    }
                }
            }
        }
    });
}

// Update species table
function updateSpeciesTable(speciesData) {
    const tbody = document.getElementById('species-table-body');
    tbody.innerHTML = '';
    
    speciesData.sort((a, b) => b.count - a.count).forEach(species => {
        const row = tbody.insertRow();
        row.insertCell(0).textContent = species.name;
        row.insertCell(1).textContent = species.count;
    });
}

// Launch ecosystem analysis
async function launchEcosystemAnalysis(type) {
    if (!window.currentGridBounds) return;
    
    const bounds = window.currentGridBounds;
    
    try {
        showLoading(true);
        
        // Set the geographic bounds in the 3D view
        geographicBounds = bounds;
        
        // Update the filter display
        const minLat = bounds.south;
        const maxLat = bounds.north;
        const minLon = bounds.west;
        const maxLon = bounds.east;
        
        document.getElementById('filter-value').textContent = 
            `${minLat.toFixed(2)}°N to ${maxLat.toFixed(2)}°N, ${minLon.toFixed(2)}°E to ${maxLon.toFixed(2)}°E`;
        document.getElementById('clear-filter').style.display = 'inline-block';
        document.getElementById('vision-controls').style.display = 'block';
        
        // Also set the input values
        document.getElementById('min-lat').value = minLat;
        document.getElementById('max-lat').value = maxLat;
        document.getElementById('min-lon').value = minLon;
        document.getElementById('max-lon').value = maxLon;
        
        const response = await fetch(
            `/api/ecosystem_analysis?north=${bounds.north}&south=${bounds.south}&east=${bounds.east}&west=${bounds.west}&type=${type}`
        );
        const data = await response.json();
        
        if (data.error) {
            alert(data.error);
            showLoading(false);
            return;
        }
        
        // Switch to ecological view - this will default to language embeddings
        switchView('ecological');
        
        // Only override if specifically requesting vision
        if (type === 'vision') {
            // Wait a bit for the view switch to complete
            setTimeout(() => {
                selectEmbeddingType('vision');
                displayVisionEmbeddings(data.embeddings);
            }, 100);
        }
        // Language will already be loaded by switchView
        
        showLoading(false);
    } catch (error) {
        console.error('Error launching ecosystem analysis:', error);
        showLoading(false);
    }
}

// Initialize 3D view
function initialize3DView() {
    const container = document.getElementById('embedding-3d');
    
    // Scene setup
    scene3D = new THREE.Scene();
    scene3D.background = new THREE.Color(0x0f172a);
    
    // Camera setup
    const aspect = container.clientWidth / container.clientHeight;
    camera3D = new THREE.PerspectiveCamera(60, aspect, 0.01, 1000);
    camera3D.position.set(3, 3, 3);
    
    // Renderer setup
    renderer3D = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer3D.setPixelRatio(window.devicePixelRatio);
    renderer3D.setSize(container.clientWidth, container.clientHeight);
    renderer3D.shadowMap.enabled = true;
    renderer3D.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer3D.domElement);
    
    // Enhanced lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    scene3D.add(ambientLight);
    
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    scene3D.add(directionalLight);
    
    // Accent lights for depth
    const accentLight1 = new THREE.DirectionalLight(0x4fc3f7, 0.2);
    accentLight1.position.set(-20, 10, -20);
    scene3D.add(accentLight1);
    
    const accentLight2 = new THREE.DirectionalLight(0xff6b6b, 0.2);
    accentLight2.position.set(20, 10, 20);
    scene3D.add(accentLight2);
    
    // Add controls - store globally for access
    window.controls3D = new THREE.OrbitControls(camera3D, renderer3D.domElement);
    window.controls3D.enableDamping = true;
    window.controls3D.dampingFactor = 0.05;
    window.controls3D.autoRotate = false;
    window.controls3D.autoRotateSpeed = 0.5;
    window.controls3D.minDistance = 0.5;
    window.controls3D.maxDistance = 100;
    window.controls3D.screenSpacePanning = false; // Keep panning in plane
    window.controls3D.mouseButtons = {
        LEFT: THREE.MOUSE.ROTATE,
        MIDDLE: THREE.MOUSE.DOLLY,
        RIGHT: THREE.MOUSE.PAN
    };
    window.controls3D.touches = {
        ONE: THREE.TOUCH.ROTATE,
        TWO: THREE.TOUCH.DOLLY_PAN
    };
    
    // Add axes helper (can be toggled)
    const axesHelper = new THREE.AxesHelper(5);
    axesHelper.visible = false; // Start hidden
    scene3D.add(axesHelper);
    window.axesHelper3D = axesHelper;
    
    // Animation loop
    function animate() {
        requestAnimationFrame(animate);
        window.controls3D.update();
        renderer3D.render(scene3D, camera3D);
    }
    animate();
    
    // Handle resize
    window.addEventListener('resize', () => {
        const width = container.clientWidth;
        const height = container.clientHeight;
        camera3D.aspect = width / height;
        camera3D.updateProjectionMatrix();
        renderer3D.setSize(width, height);
    });
}

// Geographic filter functions
function toggleFilterInputs() {
    const filterInputs = document.querySelector('.filter-inputs');
    filterInputs.style.display = filterInputs.style.display === 'none' ? 'block' : 'none';
    
    // If opening, set default values
    if (filterInputs.style.display === 'block') {
        // Use grid bounds if available, otherwise use dataset bounds
        if (window.currentGridBounds) {
            document.getElementById('min-lat').value = window.currentGridBounds.south.toFixed(4);
            document.getElementById('max-lat').value = window.currentGridBounds.north.toFixed(4);
            document.getElementById('min-lon').value = window.currentGridBounds.west.toFixed(4);
            document.getElementById('max-lon').value = window.currentGridBounds.east.toFixed(4);
        } else {
            // Set Florida dataset bounds as defaults
            document.getElementById('min-lat').value = '27.5';
            document.getElementById('max-lat').value = '29.5';
            document.getElementById('min-lon').value = '-82.0';
            document.getElementById('max-lon').value = '-80.0';
        }
    }
}

function applyGeographicFilter() {
    const minLat = parseFloat(document.getElementById('min-lat').value);
    const maxLat = parseFloat(document.getElementById('max-lat').value);
    const minLon = parseFloat(document.getElementById('min-lon').value);
    const maxLon = parseFloat(document.getElementById('max-lon').value);
    
    if (isNaN(minLat) || isNaN(maxLat) || isNaN(minLon) || isNaN(maxLon)) {
        alert('Please enter valid coordinates');
        return;
    }
    
    geographicBounds = {
        north: maxLat,
        south: minLat,
        east: maxLon,
        west: minLon
    };
    
    // Update UI
    document.getElementById('filter-value').textContent = 
        `${minLat.toFixed(2)}°N to ${maxLat.toFixed(2)}°N, ${minLon.toFixed(2)}°E to ${maxLon.toFixed(2)}°E`;
    document.getElementById('clear-filter').style.display = 'inline-block';
    document.querySelector('.filter-inputs').style.display = 'none';
    
    // Show vision controls
    document.getElementById('vision-controls').style.display = 'block';
    
    // If in language view, reload with filter
    if (currentEmbeddingView === 'language') {
        loadLanguageEmbeddings();
    }
}

// Recompute language UMAP with current geographic bounds
function recomputeLanguageUMAP() {
    loadLanguageEmbeddings(window.lastSelectedSpecies, true);
}

// Recompute vision UMAP
function recomputeVisionUMAP() {
    computeVisionUMAP();
}

// Toggle between language and vision embeddings
// Select embedding type
function selectEmbeddingType(type) {
    console.log('selectEmbeddingType called with:', type, 'current:', currentEmbeddingView);
    
    // Update button states
    document.querySelectorAll('.embedding-type-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === type);
    });
    
    // Update vision controls visibility
    const visionControls = document.getElementById('vision-controls');
    visionControls.style.display = type === 'vision' ? 'block' : 'none';
    
    // Update recompute button visibility
    const recomputeControl = document.getElementById('recompute-control');
    if (recomputeControl) {
        recomputeControl.style.display = type === 'vision' ? 'block' : 'none';
    }
    
    // Load appropriate embeddings
    currentEmbeddingView = type;
    console.log('Set currentEmbeddingView to:', type);
    
    if (type === 'vision') {
        // Check if we have non-default filters
        const hasNonDefaultFilters = window.filterState && window.filterState.hasNonDefaultFilters();
        
        if (hasNonDefaultFilters) {
            // Filters active, preload then compute
            preloadAvailableVisionEmbeddings().then(() => {
                computeVisionUMAP();
            });
        } else {
            // No filters, compute directly (will use cache if available)
            computeVisionUMAP();
        }
    } else {
        console.log('Loading language embeddings...');
        loadLanguageEmbeddings();
    }
}

// Recompute current embedding
function recomputeCurrentEmbedding() {
    if (currentEmbeddingView === 'vision') {
        recomputeVisionUMAP();
    } else {
        // For language, just refilter the existing precomputed data
        loadLanguageEmbeddings();
    }
}

// Legacy function for compatibility
function toggleEmbeddingView() {
    selectEmbeddingType(currentEmbeddingView === 'language' ? 'vision' : 'language');
}

function clearGeographicFilter() {
    geographicBounds = null;
    document.getElementById('filter-value').textContent = 'All Data';
    document.getElementById('clear-filter').style.display = 'none';
    
    // Clear input values
    document.getElementById('min-lat').value = '';
    document.getElementById('max-lat').value = '';
    document.getElementById('min-lon').value = '';
    document.getElementById('max-lon').value = '';
    
    // Clear filter state
    if (window.filterState) {
        window.filterState.clearGeographicFilter();
    }
    
    // Reload appropriate embeddings
    if (currentEmbeddingView === 'language') {
        loadLanguageEmbeddings();
    } else if (currentEmbeddingView === 'vision') {
        // Clear preloaded data
        availableVisionEmbeddings = null;
        preloadAvailableVisionEmbeddings().then(() => {
            computeVisionUMAP();
        });
    }
}

// Global storage for precomputed embeddings
let precomputedLanguageEmbeddings = null;
let precomputedVisionEmbeddings = null; // Cache for unfiltered vision UMAP

// Load language embeddings
async function loadLanguageEmbeddings(focusSpecies = null, forceRecompute = false) {
    try {
        showLoading(true);
        
        // Always load precomputed data if not already loaded
        if (!precomputedLanguageEmbeddings || forceRecompute) {
            const params = new URLSearchParams();
            params.append('precomputed', 'true');
            
            if (forceRecompute) {
                params.append('force_recompute', 'true');
            }
            
            const response = await fetch(`/api/language_embeddings/umap?${params.toString()}`);
            const data = await response.json();
            
            if (data.error) {
                console.error('Error loading precomputed language embeddings:', data.error);
                showLoading(false);
                return;
            }
            
            // Store precomputed data
            precomputedLanguageEmbeddings = data;
            console.log('Loaded precomputed language embeddings:', data.total, 'species');
        }
        
        // Now filter the precomputed data based on current filters
        let filteredEmbeddings = precomputedLanguageEmbeddings.embeddings;
        
        // Check if we have filters
        const hasFilters = window.filterState && (
            window.filterState.state.geographic ||
            window.filterState.state.temporal.yearMin !== 2010 ||
            window.filterState.state.temporal.yearMax !== 2025 ||
            window.filterState.state.temporal.monthMin !== 1 ||
            window.filterState.state.temporal.monthMax !== 12 ||
            window.filterState.state.temporal.hourMin !== 0 ||
            window.filterState.state.temporal.hourMax !== 23
        );
        
        if (hasFilters && observationsData && observationsData.length > 0) {
            // Get taxon IDs that match the current filters
            const filteredObs = observationsData.filter(obs => {
                // Apply temporal filters
                if (window.filterState) {
                    const temporal = window.filterState.state.temporal;
                    if (obs.year < temporal.yearMin || obs.year > temporal.yearMax) return false;
                    if (obs.month && (obs.month < temporal.monthMin || obs.month > temporal.monthMax)) return false;
                    if (obs.hour !== null && obs.hour !== undefined && 
                        (obs.hour < temporal.hourMin || obs.hour > temporal.hourMax)) return false;
                }
                
                // Apply geographic filters
                if (window.filterState && window.filterState.state.geographic) {
                    const bounds = window.filterState.state.geographic;
                    if (obs.lat < bounds.south || obs.lat > bounds.north) return false;
                    if (obs.lon < bounds.west || obs.lon > bounds.east) return false;
                }
                
                return true;
            });
            
            // Get unique taxon IDs from filtered observations
            const validTaxonIds = new Set(filteredObs.map(obs => obs.taxon_id));
            
            // Filter embeddings to only include valid taxon IDs
            filteredEmbeddings = precomputedLanguageEmbeddings.embeddings.filter(emb => 
                validTaxonIds.has(emb.taxon_id)
            );
            
            console.log(`Filtered to ${filteredEmbeddings.length} species from ${precomputedLanguageEmbeddings.embeddings.length}`);
        }
        
        // Display the filtered embeddings
        displayLanguageEmbeddings({
            embeddings: filteredEmbeddings,
            clusters: precomputedLanguageEmbeddings.clusters,
            total: filteredEmbeddings.length,
            precomputed: true
        });
        
        showLoading(false);
        
        // Force render update
        if (renderer3D) {
            renderer3D.render(scene3D, camera3D);
            if (window.controls3D) {
                window.controls3D.update();
            }
        }
    } catch (error) {
        console.error('Error loading language embeddings:', error);
        showLoading(false);
    }
}

// Calculate bounds for data points
function calculateBounds(points, scale = 1) {
    const bounds = {
        min: { x: Infinity, y: Infinity, z: Infinity },
        max: { x: -Infinity, y: -Infinity, z: -Infinity }
    };
    
    points.forEach(p => {
        bounds.min.x = Math.min(bounds.min.x, p.x * scale);
        bounds.min.y = Math.min(bounds.min.y, p.y * scale);
        bounds.min.z = Math.min(bounds.min.z, p.z * scale);
        bounds.max.x = Math.max(bounds.max.x, p.x * scale);
        bounds.max.y = Math.max(bounds.max.y, p.y * scale);
        bounds.max.z = Math.max(bounds.max.z, p.z * scale);
    });
    
    return bounds;
}

// Calculate center of data points
function calculateCenter(points, scale = 1) {
    const center = { x: 0, y: 0, z: 0 };
    points.forEach(p => {
        center.x += p.x * scale;
        center.y += p.y * scale;
        center.z += p.z * scale;
    });
    center.x /= points.length;
    center.y /= points.length;
    center.z /= points.length;
    return center;
}

// Display language embeddings in 3D
function displayLanguageEmbeddings(data) {
    // Extract embeddings and clusters
    const embeddings = data.embeddings || data;
    const clusters = data.clusters;
    
    currentEmbeddingView = 'language';
    
    // Update statistics
    updateEmbeddingsStatistics({
        type: 'language',
        total_points: embeddings.length,
        clusters: clusters ? Object.keys(clusters).length - (clusters['-1'] ? 1 : 0) : 0
    });
    // Update embedding type buttons to show language is active
    document.querySelectorAll('.embedding-type-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === 'language');
    });
    
    // Store for later use
    languageEmbeddings = embeddings;
    // Clear existing points
    embeddingPoints.forEach(point => {
        scene3D.remove(point.mesh);
        if (point.label) scene3D.remove(point.label);
    });
    embeddingPoints = [];
    
    // Calculate bounds and center with scaling
    const scale = 3; // Same scale used for positioning points
    const bounds = calculateBounds(embeddings, scale);
    const center = calculateCenter(embeddings, scale);
    
    // Position camera optimally
    const maxDimension = Math.max(
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z
    );
    
    const distance = maxDimension * 1.5; // Like original implementation
    camera3D.position.set(
        center.x + distance * 0.5,
        center.y + distance * 0.3,
        center.z + distance * 0.5
    );
    
    // Update controls target to center of data
    window.controls3D.target.copy(new THREE.Vector3(center.x, center.y, center.z));
    window.controls3D.update();
    camera3D.lookAt(center.x, center.y, center.z);
    
    // Create points for each embedding
    embeddings.forEach(emb => {
        // Create sphere with enhanced material
        const geometry = new THREE.SphereGeometry(0.05, 24, 24);
        
        // Use cluster color if available
        const color = emb.color ? new THREE.Color(emb.color) : new THREE.Color(0x2563eb);
        
        const material = new THREE.MeshPhysicalMaterial({ 
            color: color,
            metalness: 0.1,
            roughness: 0.4,
            clearcoat: 0.3,
            clearcoatRoughness: 0.4,
            emissive: color,
            emissiveIntensity: 0.05
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(emb.x * 3, emb.y * 3, emb.z * 3);
        sphere.castShadow = true;
        sphere.receiveShadow = true;
        
        // Store data
        sphere.userData = emb;
        
        // Add to scene
        scene3D.add(sphere);
        
        // Create label with italic font
        if (document.getElementById('show-labels').checked) {
            const label = createTextLabel(emb.name, sphere.position, true);
            scene3D.add(label);
            embeddingPoints.push({ mesh: sphere, label: label, data: emb, type: 'language' });
        } else {
            embeddingPoints.push({ mesh: sphere, data: emb, type: 'language' });
        }
        
        // Add click handler
        sphere.callback = () => {
            showPointInfo(emb, 'language');
            flyToObject(sphere);
        };
    });
    
    // Add raycaster for mouse interaction
    setupRaycaster();
}

// Update statistics in embeddings view
function updateEmbeddingsStatistics(stats) {
    if (currentView !== 'ecological') return;
    
    // Update the main statistics panel
    if (stats.type === 'language') {
        document.getElementById('total-species').textContent = stats.total_points || '-';
        document.getElementById('total-observations').textContent = '-';
        document.getElementById('total-images').textContent = '-';
    } else if (stats.type === 'vision') {
        document.getElementById('total-species').textContent = stats.unique_species || '-';
        document.getElementById('total-observations').textContent = stats.total_points || '-';
        document.getElementById('total-images').textContent = stats.total_points || '-';
    }
}

// Display vision embeddings in 3D
function displayVisionEmbeddings(embeddings) {
    // Update statistics
    const uniqueSpecies = new Set(embeddings.map(e => e.taxon_id)).size;
    updateEmbeddingsStatistics({
        type: 'vision',
        total_points: embeddings.length,
        unique_species: uniqueSpecies
    });
    // Clear existing points
    embeddingPoints.forEach(point => {
        scene3D.remove(point.mesh);
        if (point.label) scene3D.remove(point.label);
    });
    embeddingPoints = [];
    
    // Calculate bounds and center with scaling
    const scale = 3; // Same scale used for positioning points
    const bounds = calculateBounds(embeddings, scale);
    const center = calculateCenter(embeddings, scale);
    
    // Position camera optimally
    const maxDimension = Math.max(
        bounds.max.x - bounds.min.x,
        bounds.max.y - bounds.min.y,
        bounds.max.z - bounds.min.z
    );
    
    const distance = maxDimension * 1.5; // Like original implementation
    camera3D.position.set(
        center.x + distance * 0.5,
        center.y + distance * 0.3,
        center.z + distance * 0.5
    );
    
    // Update controls target to center of data
    window.controls3D.target.copy(new THREE.Vector3(center.x, center.y, center.z));
    window.controls3D.update();
    camera3D.lookAt(center.x, center.y, center.z);
    
    // Create points with species-specific colors
    embeddings.forEach(emb => {
        // Create sphere with enhanced material
        const geometry = new THREE.SphereGeometry(0.05, 24, 24);
        const color = new THREE.Color(emb.color);
        const material = new THREE.MeshPhysicalMaterial({ 
            color: color,
            metalness: 0.1,
            roughness: 0.4,
            clearcoat: 0.3,
            clearcoatRoughness: 0.4,
            emissive: color,
            emissiveIntensity: 0.05
        });
        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(emb.x * 3, emb.y * 3, emb.z * 3);
        sphere.castShadow = true;
        sphere.receiveShadow = true;
        
        // Store data
        sphere.userData = emb;
        
        // Add to scene
        scene3D.add(sphere);
        
        // Create label with species name
        if (document.getElementById('show-labels').checked) {
            const label = createTextLabel(emb.taxon_name, sphere.position, true);
            scene3D.add(label);
            embeddingPoints.push({ mesh: sphere, label: label, data: emb, type: 'vision' });
        } else {
            embeddingPoints.push({ mesh: sphere, data: emb, type: 'vision' });
        }
        
        // Add click handler
        sphere.callback = () => {
            showPointInfo(emb, 'vision');
            flyToObject(sphere);
        };
    });
    
    setupRaycaster();
}

// Create text label
function createTextLabel(text, position, italic = false) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 256;
    canvas.height = 64;
    
    context.fillStyle = '#f1f5f9';
    context.font = italic ? 'italic 24px Inter' : '24px Inter';
    context.fillText(text, 10, 40);
    
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, opacity: 0.9 });
    const sprite = new THREE.Sprite(material);
    
    sprite.position.copy(position);
    sprite.position.y += 0.1;
    sprite.scale.set(1, 0.25, 1);
    
    return sprite;
}

// Setup raycaster for mouse interaction
function setupRaycaster() {
    const raycaster = new THREE.Raycaster();
    const mouse = new THREE.Vector2();
    
    renderer3D.domElement.addEventListener('click', onMouseClick);
    
    function onMouseClick(event) {
        const rect = renderer3D.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        
        raycaster.setFromCamera(mouse, camera3D);
        
        const meshes = embeddingPoints.map(p => p.mesh);
        const intersects = raycaster.intersectObjects(meshes);
        
        if (intersects.length > 0) {
            const clicked = intersects[0].object;
            if (clicked.callback) {
                clicked.callback();
            }
        }
    }
}

// Preload available vision embeddings
async function preloadAvailableVisionEmbeddings() {
    if (isPreloadingVision || availableVisionEmbeddings) return;
    
    // Only preload if we have non-default filters
    const hasNonDefaultFilters = window.filterState && window.filterState.hasNonDefaultFilters();
    if (!hasNonDefaultFilters) {
        console.log('No filters active, skipping vision embeddings preload');
        return;
    }
    
    isPreloadingVision = true;
    console.log('Preloading available vision embeddings for filtered data...');
    
    try {
        // Get current filters
        const filters = window.filterState ? window.filterState.getFilters() : null;
        
        const maxImages = parseInt(document.getElementById('max-images-slider')?.value || 250);
        
        // Build URL with filter state parameters
        const params = window.filterState ? window.filterState.getAPIParams() : new URLSearchParams();
        params.append('max_images', maxImages);
        
        const response = await fetch(`/api/vision_embeddings/available?${params.toString()}`);
        const data = await response.json();
        
        if (data.observations) {
            availableVisionEmbeddings = data.observations;
            console.log(`Preloaded ${data.count} observations with vision embeddings`);
            
            // Update UI to show readiness
            const recomputeBtn = document.getElementById('recompute-btn');
            if (recomputeBtn && currentEmbeddingView === 'vision') {
                recomputeBtn.classList.add('ready');
                recomputeBtn.title = `${data.count} images ready for UMAP computation`;
            }
        }
    } catch (error) {
        console.error('Error preloading vision embeddings:', error);
    } finally {
        isPreloadingVision = false;
    }
}

// Compute Vision UMAP
async function computeVisionUMAP() {
    console.log('computeVisionUMAP called, currentEmbeddingView:', currentEmbeddingView);
    
    // Don't compute vision if we're not in vision mode
    if (currentEmbeddingView !== 'vision') {
        console.log('Not in vision mode, aborting computeVisionUMAP');
        return;
    }
    
    const maxImages = parseInt(document.getElementById('max-images-slider').value);
    
    // Check if we have non-default filters
    const hasNonDefaultFilters = window.filterState && window.filterState.hasNonDefaultFilters();
    
    // If no filters and we have cached unfiltered data, use it
    if (!hasNonDefaultFilters && precomputedVisionEmbeddings) {
        console.log('Using cached unfiltered vision UMAP');
        displayVisionEmbeddings(precomputedVisionEmbeddings.embeddings);
        return;
    }
    
    // Show loading indicator on button
    const recomputeBtn = document.getElementById('recompute-btn');
    if (recomputeBtn) {
        recomputeBtn.disabled = true;
        recomputeBtn.innerHTML = '<span class="icon">⏳</span> Computing Vision UMAP...';
    }
    
    // Show progress modal
    showVisionProgress('Initializing Vision V-JEPA 2 computation...', 10);
    
    try {
        // Build URL with filter state parameters
        const params = window.filterState ? window.filterState.getAPIParams() : new URLSearchParams();
        params.append('max_images', maxImages);
        
        // If we have preloaded data, pass the specific GBIFs to use
        if (availableVisionEmbeddings && availableVisionEmbeddings.length > 0) {
            // Add the preloaded GBIF IDs to ensure we only compute for available embeddings
            const gbifIds = availableVisionEmbeddings.map(obs => obs.gbif_id).slice(0, maxImages);
            params.append('gbif_ids', gbifIds.join(','));
        }
        
        const url = `/api/vision_embeddings/umap?${params.toString()}`;
        
        // Update progress
        showVisionProgress('Loading vision embeddings from dataset...', 30);
            
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        if (data.error) {
            hideVisionProgress();
            alert(data.error);
            return;
        }
        
        // Update progress with embedding count
        showVisionProgress(`Processing ${data.embeddings.length} vision embeddings...`, 60);
        
        // Store vision embeddings
        visionEmbeddings = data.embeddings;
        
        // Cache unfiltered results
        if (!hasNonDefaultFilters) {
            precomputedVisionEmbeddings = {
                embeddings: data.embeddings,
                total: data.total,
                bounds: data.bounds
            };
            console.log('Cached unfiltered vision UMAP with', data.embeddings.length, 'points');
        }
        
        // Display vision embeddings
        // Note: Don't override currentEmbeddingView here - it should be set by selectEmbeddingType
        
        // Only display if we're still in vision mode (user might have switched away)
        if (currentEmbeddingView === 'vision') {
            // Update progress
            showVisionProgress('Rendering 3D visualization...', 90);
            
            // Update embedding type buttons to show vision is active
            document.querySelectorAll('.embedding-type-btn').forEach(btn => {
                btn.classList.toggle('active', btn.dataset.type === 'vision');
            });
            document.getElementById('vision-controls').style.display = 'block';
            displayVisionEmbeddings(data.embeddings);
            
            // Hide progress after a brief moment
            setTimeout(() => {
                hideVisionProgress();
            }, 500);
        } else {
            console.log('User switched away from vision mode, not displaying vision embeddings');
            hideVisionProgress();
        }
        
        // Force render update
        if (renderer3D) {
            renderer3D.render(scene3D, camera3D);
            if (window.controls3D) {
                window.controls3D.update();
            }
        }
        
    } catch (error) {
        console.error('Error computing vision UMAP:', error);
        hideVisionProgress();
        alert('Failed to compute vision UMAP: ' + error.message);
    } finally {
        // Reset button state
        const recomputeBtn = document.getElementById('recompute-btn');
        if (recomputeBtn) {
            recomputeBtn.disabled = false;
            recomputeBtn.innerHTML = '<span class="icon">🔄</span> Recompute UMAP';
        }
    }
}

// Animate from vision to language embeddings
function animateVisionToLanguage() {
    currentEmbeddingView = 'language';
    // Update embedding type buttons to show language is active
    document.querySelectorAll('.embedding-type-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === 'language');
    });
    document.getElementById('vision-controls').style.display = 'none';
    
    // Clear vision embeddings
    embeddingPoints.forEach(point => {
        scene3D.remove(point.mesh);
        if (point.label) scene3D.remove(point.label);
    });
    embeddingPoints = [];
    
    // Reload language embeddings
    loadLanguageEmbeddings(window.lastSelectedSpecies);
}

// Animate from language to vision embeddings
function animateLanguageToVision(visionData) {
    currentEmbeddingView = 'vision';
    // Update embedding type buttons to show vision is active
    document.querySelectorAll('.embedding-type-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.type === 'vision');
    });
    
    // Group vision embeddings by taxon
    const visionByTaxon = {};
    visionData.forEach(v => {
        if (!visionByTaxon[v.taxon_id]) {
            visionByTaxon[v.taxon_id] = [];
        }
        visionByTaxon[v.taxon_id].push(v);
    });
    
    // Animate burst effect
    const newPoints = [];
    let totalDelay = 0;
    
    Object.entries(visionByTaxon).forEach(([taxonId, visionPoints]) => {
        // Find original language point
        const langPoint = embeddingPoints.find(p => p.data.taxon_id === taxonId);
        if (!langPoint) return;
        
        visionPoints.forEach((visionEmb, index) => {
            setTimeout(() => {
                // Create new sphere for vision embedding
                const geometry = new THREE.SphereGeometry(0.04, 24, 24);
                const color = new THREE.Color(visionEmb.color);
                const material = new THREE.MeshPhysicalMaterial({
                    color: color,
                    metalness: 0.1,
                    roughness: 0.4,
                    clearcoat: 0.3,
                    clearcoatRoughness: 0.4,
                    emissive: color,
                    emissiveIntensity: 0.05,
                    opacity: 0,
                    transparent: true
                });
                
                const sphere = new THREE.Mesh(geometry, material);
                sphere.position.copy(langPoint.mesh.position);
                sphere.userData = visionEmb;
                scene3D.add(sphere);
                
                // Create bezier curve for animation
                const start = langPoint.mesh.position.clone();
                const end = new THREE.Vector3(visionEmb.x * 3, visionEmb.y * 3, visionEmb.z * 3);
                const control = createControlPoint(start, end, animationParams.loopiness);
                
                // Animate along curve
                animateSphereAlongCurve(sphere, start, control, end, () => {
                    // Add click handler
                    sphere.callback = () => {
                        showPointInfo(visionEmb, 'vision');
                        flyToObject(sphere);
                    };
                });
                
                newPoints.push({ mesh: sphere, data: visionEmb, type: 'vision' });
                
            }, totalDelay);
            
            totalDelay += animationParams.stagger;
        });
    });
    
    // Fade out language points after animation starts
    setTimeout(() => {
        embeddingPoints.forEach(point => {
            if (point.type === 'language') {
                fadeOutObject(point.mesh);
                if (point.label) fadeOutObject(point.label);
            }
        });
    }, animationParams.speed * 0.3);
    
    // Update points array after animation
    setTimeout(() => {
        embeddingPoints = newPoints;
        setupRaycaster();
    }, animationParams.speed + totalDelay);
}

// Create control point for bezier curve
function createControlPoint(start, end, loopiness) {
    const mid = start.clone().add(end).multiplyScalar(0.5);
    const offset = new THREE.Vector3(
        (Math.random() - 0.5) * loopiness * 5,
        Math.abs(Math.random() * loopiness * 5),
        (Math.random() - 0.5) * loopiness * 5
    );
    return mid.add(offset);
}

// Animate sphere along bezier curve
function animateSphereAlongCurve(sphere, start, control, end, onComplete) {
    const curve = new THREE.QuadraticBezierCurve3(start, control, end);
    const startTime = Date.now();
    
    function animate() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / animationParams.speed, 1);
        
        // Easing
        const eased = 1 - Math.pow(1 - progress, 3);
        
        // Update position
        const position = curve.getPoint(eased);
        sphere.position.copy(position);
        
        // Update opacity
        sphere.material.opacity = eased;
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            sphere.material.transparent = false;
            sphere.material.opacity = 1;
            if (onComplete) onComplete();
        }
    }
    
    animate();
}

// Fade out object
function fadeOutObject(object) {
    const startOpacity = object.material.opacity;
    const startTime = Date.now();
    const duration = animationParams.speed * 0.5;
    
    object.material.transparent = true;
    
    function animate() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        object.material.opacity = startOpacity * (1 - progress);
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        } else {
            scene3D.remove(object);
        }
    }
    
    animate();
}

// Switch back to language view
function switchToLanguageView() {
    if (currentEmbeddingView === 'language') return;
    
    // Animate vision points back to language positions
    const pointsByTaxon = {};
    embeddingPoints.forEach(p => {
        if (!pointsByTaxon[p.data.taxon_id]) {
            pointsByTaxon[p.data.taxon_id] = [];
        }
        pointsByTaxon[p.data.taxon_id].push(p);
    });
    
    let totalDelay = 0;
    
    Object.entries(pointsByTaxon).forEach(([taxonId, points]) => {
        // Find language position
        const langEmb = languageEmbeddings.find(l => l.taxon_id === taxonId);
        if (!langEmb) return;
        
        const langPos = new THREE.Vector3(langEmb.x * 3, langEmb.y * 3, langEmb.z * 3);
        
        points.forEach((point, index) => {
            setTimeout(() => {
                const start = point.mesh.position.clone();
                const control = createControlPoint(start, langPos, animationParams.loopiness * 0.5);
                
                animateSphereAlongCurve(point.mesh, start, control, langPos, () => {
                    // Remove after reaching destination
                    fadeOutObject(point.mesh);
                });
            }, totalDelay);
            
            totalDelay += animationParams.stagger;
        });
    });
    
    // Recreate language view after animation
    setTimeout(() => {
        displayLanguageEmbeddings({ embeddings: languageEmbeddings });
    }, animationParams.speed + totalDelay);
}

// Enable debug mode
function enableDebugMode() {
    debugMode = true;
    document.getElementById('debug-controls').style.display = 'block';
    console.log('Debug mode enabled');
}

// Show point info (unified for both language and vision)
function showPointInfo(data, type) {
    currentPointData = { data, type };
    
    const panel = document.getElementById('point-info-panel');
    const title = document.getElementById('point-title');
    const details = document.getElementById('point-details');
    
    if (type === 'language') {
        // Language embedding point (species)
        title.textContent = data.name;
        details.innerHTML = `
            <div class="detail-item">
                <span class="label">Taxon ID:</span>
                <span class="value">${data.taxon_id}</span>
            </div>
            <div class="detail-item">
                <span class="label">Observations:</span>
                <span class="value">${data.count}</span>
            </div>
            <div class="detail-item">
                <span class="label">Cluster:</span>
                <span class="value" style="background: ${data.color}; padding: 2px 8px; border-radius: 4px; color: white;">
                    ${data.cluster === -1 ? 'Noise' : `Cluster ${data.cluster}`}
                </span>
            </div>
        `;
        
        // Load all images for this species
        loadSpeciesImages(data.taxon_id);
        
    } else {
        // Vision embedding point (individual observation)
        title.textContent = data.taxon_name;
        details.innerHTML = `
            <div class="detail-item">
                <span class="label">GBIF ID:</span>
                <span class="value">${data.gbif_id}</span>
            </div>
            <div class="detail-item">
                <span class="label">Location:</span>
                <span class="value">${data.lat.toFixed(4)}°N, ${data.lon.toFixed(4)}°E</span>
            </div>
        `;
        
        // Load single observation images
        loadObservationImages(data.gbif_id);
    }
    
    panel.style.display = 'block';
}

// Load species images
async function loadSpeciesImages(taxonId) {
    try {
        // Show loading state
        const gallery = document.getElementById('point-image-gallery');
        gallery.style.display = 'block';
        document.getElementById('gallery-image').src = '';
        document.getElementById('gallery-image').alt = 'Loading...';
        
        // Use efficient species-specific endpoint
        const speciesResponse = await fetch(`/api/species/${taxonId}/observations`);
        const speciesData = await speciesResponse.json();
        
        if (speciesData.observations_with_vision === 0) {
            gallery.style.display = 'none';
            return;
        }
        
        console.log(`Loading images for species ${speciesData.taxon_name}: ${speciesData.observations.length} of ${speciesData.observations_with_vision} observations with vision`);
        
        if (speciesData.truncated) {
            console.log(`Note: Results limited to ${speciesData.max_returned} observations for performance`);
        }
        
        // Store observation list for on-demand loading
        galleryObservations = speciesData.observations;
        galleryImages = new Array(galleryObservations.length).fill(null);
        galleryIndex = 0;
        
        // Load only the first image immediately
        await loadGalleryImage(0);
        
        // Update display
        updateGalleryDisplay();
        document.getElementById('point-image-gallery').style.display = 'block';
        document.getElementById('vision-feature-panel').style.display = 'block';
        
        // Initialize gallery feature controls
        galleryCurrentVisualization = 'pca1';
        document.getElementById('galleryVisualizationMethod').value = 'pca1';
        galleryIsUMAPActive = false;
        document.getElementById('galleryUmapBtnText').textContent = 'Show UMAP RGB';
        
        // Preload next few images in background
        if (galleryObservations.length > 1) {
            for (let i = 1; i <= Math.min(3, galleryObservations.length - 1); i++) {
                loadGalleryImage(i); // Don't await - load in background
            }
        }
        
    } catch (error) {
        console.error('Error loading species images:', error);
    }
}

// Load observation images
async function loadObservationImages(gbifId) {
    try {
        const response = await fetch(`/api/observation/${gbifId}`);
        const data = await response.json();
        
        if (data.images && data.images.length > 0) {
            galleryImages = data.images.map(img => ({
                ...img,
                gbif_id: gbifId
            }));
            galleryIndex = 0;
            updateGalleryDisplay();
            document.getElementById('point-image-gallery').style.display = 'block';
            document.getElementById('vision-feature-panel').style.display = 'block';
        } else {
            document.getElementById('point-image-gallery').style.display = 'none';
        }
        
    } catch (error) {
        console.error('Error loading observation images:', error);
    }
}

// Update gallery display
function updateGalleryDisplay() {
    if (!galleryObservations || galleryObservations.length === 0) return;
    
    const currentImage = galleryImages[galleryIndex];
    const imgElement = document.getElementById('gallery-image');
    
    if (!currentImage) {
        // Image not loaded yet
        imgElement.src = '';
        imgElement.alt = 'Loading...';
        imgElement.style.opacity = '0.5';
    } else if (currentImage.error) {
        // Failed to load
        imgElement.src = '';
        imgElement.alt = 'Failed to load image';
        imgElement.style.opacity = '0.5';
    } else {
        // Successfully loaded
        imgElement.src = currentImage.url;
        imgElement.alt = `${currentImage.taxon_name || 'Species'} observation`;
        imgElement.style.opacity = '1';
    }
    
    document.getElementById('gallery-current').textContent = galleryIndex + 1;
    document.getElementById('gallery-total').textContent = galleryObservations.length;
    
    // Show/hide navigation
    const showNav = galleryObservations.length > 1;
    document.querySelector('.gallery-nav.prev').style.display = showNav ? 'flex' : 'none';
    document.querySelector('.gallery-nav.next').style.display = showNav ? 'flex' : 'none';
    
    // Store current image ID for vision features if loaded
    if (currentImage && !currentImage.error) {
        window.currentGalleryImageId = currentImage.image_id;
        galleryCurrentObservationId = currentImage.gbif_id || currentImage.observation_id;
        
        // Store for view switching
        window.lastSelectedGbifId = galleryCurrentObservationId;
        
        // Load image with vision features using the unified manager
        if (galleryVisionManager && currentImage.image_id) {
            galleryVisionManager.loadImage(currentImage.image_id, galleryCurrentObservationId);
        }
    }
}

// Load a single gallery image on demand
async function loadGalleryImage(index) {
    if (!galleryObservations || index >= galleryObservations.length) return;
    
    // Check if already loaded
    if (galleryImages[index] !== null) return;
    
    try {
        const obs = galleryObservations[index];
        console.log(`Loading image ${index + 1}/${galleryObservations.length} for observation ${obs.gbif_id}`);
        
        const detailResponse = await fetch(`/api/observation/${obs.gbif_id}`);
        const detail = await detailResponse.json();
        
        if (detail.images && detail.images.length > 0) {
            galleryImages[index] = {
                ...detail.images[0],
                gbif_id: obs.gbif_id,
                observation: obs,
                taxon_name: detail.taxon_name
            };
            console.log(`Successfully loaded image at index ${index}:`, galleryImages[index].url);
            
            // If this is the currently displayed image, update the display
            if (index === galleryIndex) {
                updateGalleryDisplay();
            }
        } else {
            console.log(`No images found for observation ${obs.gbif_id}`);
            galleryImages[index] = { error: true, message: 'No images available' };
        }
    } catch (error) {
        console.error(`Failed to load observation at index ${index}:`, error);
        galleryImages[index] = { error: true }; // Mark as attempted
    }
}

// Navigate gallery with on-demand loading
async function navigateGallery(direction) {
    if (!galleryObservations || galleryObservations.length === 0) return;
    if (galleryObservations.length <= 1) return;
    
    const newIndex = (galleryIndex + direction + galleryObservations.length) % galleryObservations.length;
    
    // Show loading state immediately
    const img = document.getElementById('gallery-image');
    if (img) {
        img.style.opacity = '0.5';
        img.alt = 'Loading...';
    }
    
    // Load the image if not already loaded
    await loadGalleryImage(newIndex);
    
    galleryIndex = newIndex;
    updateGalleryDisplay();
    
    // Preload next/previous images in background
    const preloadNext = (newIndex + 1) % galleryObservations.length;
    const preloadPrev = (newIndex - 1 + galleryObservations.length) % galleryObservations.length;
    loadGalleryImage(preloadNext); // Don't await - load in background
    loadGalleryImage(preloadPrev); // Don't await - load in background
}

// Close point panel
function closePointPanel() {
    document.getElementById('point-info-panel').style.display = 'none';
    currentPointData = null;
    galleryImages = [];
    galleryObservations = [];
}

// Gallery vision feature controls
let galleryTemporalMode = 'mean';
let galleryVisualization = 'l2norm';
let galleryColormap = 'plasma';
let galleryAlpha = 0.7;
let galleryVisionManager = null; // Initialized in document ready

// These old functions are replaced by the VisionFeatureManager-based versions below

// Removed duplicate - using unified function below

// Removed duplicate updateGalleryAlpha - using unified function below

// Legacy updateGalleryVisualization - now just triggers vision manager update
async function updateGalleryVisualization() {
    if (!window.currentGalleryImageId || !galleryVisionManager) return;
    
    // The vision manager handles all visualization updates
    // This function is kept for backward compatibility with old code paths
    await galleryVisionManager.updateVisualization();
}

// Gallery UMAP is now handled by the async toggleGalleryUMAP function below

// Change base layer
function changeBaseLayer(layer) {
    // Remove all tile layers
    map.eachLayer(function(l) {
        if (l instanceof L.TileLayer) {
            map.removeLayer(l);
        }
    });
    
    // Add new base layer
    let tileUrl;
    switch(layer) {
        case 'satellite':
            tileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}';
            break;
        case 'terrain':
            tileUrl = 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png';
            break;
        case 'streets':
            tileUrl = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
            break;
    }
    
    L.tileLayer(tileUrl, {
        attribution: 'Map data',
        maxZoom: 19
    }).addTo(map);
}

// Loading indicator
function showLoading(show) {
    console.log('showLoading called with:', show, 'currentView:', currentView, 'currentEmbeddingView:', currentEmbeddingView);
    document.getElementById('loading').style.display = show ? 'flex' : 'none';
}

// Vision progress modal
let visionProgressModal = null;

function showVisionProgress(message, progress = 0) {
    if (!visionProgressModal) {
        // Create modal
        visionProgressModal = document.createElement('div');
        visionProgressModal.className = 'vision-progress-modal';
        visionProgressModal.innerHTML = `
            <div class="vision-progress-content">
                <h3>Vision V-JEPA 2 Processing</h3>
                <div class="vision-progress-message"></div>
                <div class="vision-progress-bar">
                    <div class="vision-progress-fill"></div>
                </div>
                <div class="vision-progress-stats">
                    <span class="progress-percent">0%</span>
                    <span class="progress-details"></span>
                </div>
            </div>
        `;
        document.body.appendChild(visionProgressModal);
    }
    
    // Update content
    visionProgressModal.querySelector('.vision-progress-message').textContent = message;
    visionProgressModal.querySelector('.vision-progress-fill').style.width = progress + '%';
    visionProgressModal.querySelector('.progress-percent').textContent = progress + '%';
    visionProgressModal.style.display = 'flex';
}

function hideVisionProgress() {
    if (visionProgressModal) {
        visionProgressModal.style.display = 'none';
    }
}

// Close grid overlay
document.querySelector('#grid-stats-overlay .close-btn').addEventListener('click', function() {
    document.getElementById('grid-stats-overlay').style.display = 'none';
    if (selectedGrid) {
        map.removeLayer(selectedGrid);
        selectedGrid = null;
    }
});

// OrbitControls are now loaded from CDN

// Vision Feature Functions
async function loadImageAndFeatures(imageId) {
    try {
        performanceMonitor.start('loadImageAndFeatures');
        performanceMonitor.log('Starting vision features load', { imageId, currentSettings: { temporal: currentTemporalMode, colormap: currentColormap, alpha: currentAlpha, visualization: currentVisualization } });
        
        currentObservationId = imageId;
        currentPCAData = null; // Clear previous PCA data
        
        // Clear previous overlay to prevent persistence
        const overlayContainer = document.getElementById('obs-attention-overlay');
        const overlayImg = document.getElementById('obs-attention-img');
        if (overlayImg) {
            overlayImg.src = '';
            overlayImg.style.transform = 'none'; // Reset transform
        }
        if (overlayContainer) overlayContainer.style.display = 'none';
        
        // Reset container aspect ratio
        const container = document.querySelector('.image-container');
        if (container) {
            container.style.aspectRatio = 'auto';
        }
        
        // Load original image
        const img = document.getElementById('obs-image');
        
        // Extract GBIF ID and image number from imageId (format: gbif_XXXXXXX_taxon_XXXXXXX_img_N)
        const match = imageId.match(/gbif_(\d+)_taxon_\d+_img_(\d+)/);
        let actualGbifId, imageNum;
        
        if (match) {
            actualGbifId = match[1];
            imageNum = match[2];
        } else {
            // Fallback - try to extract just the GBIF ID if different format
            console.warn('Unexpected imageId format:', imageId);
            actualGbifId = imageId.replace(/[^0-9]/g, ''); // Extract numbers only
            imageNum = 1;
        }
        
        img.src = `/api/image_proxy/${actualGbifId}/${imageNum}`;
        
        // Wait for image to load to get its natural dimensions
        let imageAspectRatio = 1;
        await new Promise((resolve, reject) => {
            img.onload = () => {
                // Calculate the image aspect ratio for stretching the overlay
                imageAspectRatio = img.naturalWidth / img.naturalHeight;
                currentImageAspectRatio = imageAspectRatio; // Store globally
                resolve();
            };
            img.onerror = reject;
        });
        
        // Show loading state
        if (overlayContainer) overlayContainer.style.opacity = '0.5';
        
        // Load attention map (but don't fail if not available)
        console.log(`🔄 Loading vision features for ${imageId} with settings:`, {
            temporal: currentTemporalMode,
            colormap: currentColormap,
            alpha: currentAlpha,
            visualization: currentVisualization
        });
        
        // For PCA1, use the fast raw endpoint and generate visualization client-side
        if (currentVisualization === 'pca1' || currentVisualization.startsWith('pca')) {
            try {
                const startTime = performance.now();
                const response = await fetch(`/api/features/${imageId}/pca-raw`);
                
                if (response.ok) {
                    const data = await response.json();
                    const fetchTime = performance.now() - startTime;
                    console.log(`⚡ PCA raw data fetched in ${fetchTime.toFixed(1)}ms`);
                    
                    // Store PCA data for reuse
                    currentPCAData = data;
                    
                    // Generate overlay client-side
                    const overlayDataUrl = await generatePCAOverlay(data.pca_values, currentColormap, currentAlpha);
                    
                    if (overlayImg && overlayDataUrl) {
                        overlayImg.src = overlayDataUrl;
                        
                        // Apply aspect ratio correction
                        if (imageAspectRatio > 1) {
                            overlayImg.style.transform = `scaleX(${imageAspectRatio})`;
                        } else if (imageAspectRatio < 1) {
                            overlayImg.style.transform = `scaleY(${1 / imageAspectRatio})`;
                        } else {
                            overlayImg.style.transform = 'none';
                        }
                        
                        overlayContainer.style.display = 'block';
                        overlayContainer.style.opacity = '1';
                    }
                    
                    // Update stats
                    updateFeatureStats({
                        max_attention: data.stats.max.toFixed(3),
                        mean_attention: data.stats.mean.toFixed(3),
                        std_attention: data.stats.std.toFixed(3),
                        explained_variance: (data.stats.explained_variance * 100).toFixed(1) + '%'
                    });
                    
                    performanceMonitor.end('loadImageAndFeatures');
                    return;
                }
            } catch (error) {
                console.warn('Fast PCA endpoint failed, falling back to standard endpoint:', error);
            }
        }
        
        // Fall back to standard attention endpoint for other visualizations
        try {
            const startTime = performance.now();
            
            // Add timeout to prevent hanging requests
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 second timeout
            
            const response = await fetch(`/api/features/${imageId}/attention?temporal=${currentTemporalMode}&colormap=${currentColormap}&alpha=${currentAlpha}&visualization=${currentVisualization}`, {
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            const fetchTime = performance.now() - startTime;
            console.log(`⚡ Vision features fetch completed in ${fetchTime.toFixed(1)}ms, status: ${response.status}`);
            
            if (!response.ok) {
                console.warn(`No vision features available for ${imageId}`);
                if (overlayImg) overlayImg.src = '';
                if (overlayContainer) overlayContainer.style.display = 'none';
                
                // Update stats to show no features
                updateFeatureStats({
                    max_attention: 'N/A',
                    mean_attention: 'N/A',
                    spatial_diversity: 'N/A',
                    temporal_stability: 'N/A'
                });
                
                // Hide vision feature controls
                const featurePanel = document.querySelector('.feature-controls-panel');
                if (featurePanel) featurePanel.style.display = 'none';
                return;
            }
        
        const data = await response.json();
        
        // Auto-show overlay when vision features are loaded (as requested by user)
        const toggleOverlayCheckbox = document.getElementById('toggle-overlay');
        if (toggleOverlayCheckbox) {
            toggleOverlayCheckbox.checked = true;
            console.log(`✅ Auto-enabled overlay checkbox for vision features`);
        }
        
        const showOverlay = true; // Always show overlay when vision features are available
        console.log(`👁️ Auto-showing vision features overlay (PCA Component 1 in Plasma)`);
        if (overlayContainer) {
            overlayContainer.style.display = 'block';
            overlayContainer.style.opacity = '1';
            console.log(`📺 Overlay container displayed with vision features`);
        }
        
        console.log('📋 Response data received:', {
            mode: data.mode,
            hasAttentionMap: !!data.attention_map,
            hasStats: !!data.stats,
            dataKeys: Object.keys(data)
        });
        
        if (data.mode === 'spatial' || data.mode === 'mean') {
            if (overlayImg) {
                console.log('🖼️ Setting attention map overlay:', {
                    hasAttentionMap: !!data.attention_map,
                    attentionMapType: typeof data.attention_map,
                    attentionMapLength: data.attention_map ? data.attention_map.length : 'N/A',
                    attentionMapPrefix: data.attention_map ? data.attention_map.substring(0, 50) : 'N/A',
                    imageAspectRatio: imageAspectRatio
                });
                
                overlayImg.src = data.attention_map;
                
                // Verify the image was set
                console.log('✅ Overlay image src set to:', overlayImg.src ? overlayImg.src.substring(0, 50) + '...' : 'EMPTY');
                
                // Apply aspect ratio stretching to match the original image
                // V-JEPA processes square images, so we need to stretch the overlay
                if (imageAspectRatio > 1) {
                    // Horizontal image - stretch horizontally
                    overlayImg.style.transform = `scaleX(${imageAspectRatio})`;
                } else if (imageAspectRatio < 1) {
                    // Vertical image - stretch vertically
                    overlayImg.style.transform = `scaleY(${1 / imageAspectRatio})`;
                } else {
                    // Square image - no stretching needed
                    overlayImg.style.transform = 'none';
                }
                
                console.log('🔧 Applied transform:', overlayImg.style.transform);
            }
            updateFeatureStats(data.stats);
        } else {
            temporalFrames = data.attention_frames;
            document.getElementById('temporalSlider').style.display = 'block';
            updateTemporalFrame(0);
            // Apply the same stretching for temporal frames
            if (overlayImg) {
                if (imageAspectRatio > 1) {
                    overlayImg.style.transform = `scaleX(${imageAspectRatio})`;
                } else if (imageAspectRatio < 1) {
                    overlayImg.style.transform = `scaleY(${1 / imageAspectRatio})`;
                } else {
                    overlayImg.style.transform = 'none';
                }
            }
        }
        
        // Load detailed statistics
        loadFeatureStatistics(imageId);
        
        performanceMonitor.end('loadImageAndFeatures');
        performanceMonitor.log('Vision features loaded successfully', { imageId });
        
        } catch (error) {
            performanceMonitor.log('ERROR loading vision features', { 
                error: error.message, 
                imageId, 
                stack: error.stack,
                currentSettings: { temporal: currentTemporalMode, colormap: currentColormap, alpha: currentAlpha, visualization: currentVisualization }
            });
            
            if (error.name === 'AbortError') {
                console.error('⏱️ Vision features request timed out after 30 seconds');
                alert('Vision features computation timed out. This may be due to server load. Please try again.');
            } else {
                console.error('❌ Error loading features:', error);
            }
            
            const overlayContainer = document.getElementById('obs-attention-overlay');
            if (overlayContainer) overlayContainer.style.display = 'none';
            performanceMonitor.end('loadImageAndFeatures');
        }
        
    } catch (error) {
        performanceMonitor.log('ERROR loading image', { 
            error: error.message, 
            imageId, 
            stack: error.stack 
        });
        console.error('❌ Error loading image:', error);
        performanceMonitor.end('loadImageAndFeatures');
    }
}

// Load feature statistics
async function loadFeatureStatistics(imageId) {
    try {
        const response = await fetch(`/api/features/${imageId}/statistics`);
        if (!response.ok) {
            console.error('Failed to load statistics');
            return;
        }
        const stats = await response.json();
        
        document.getElementById('spatialDiversity').textContent = stats.spatial_diversity.toFixed(3);
        document.getElementById('temporalStability').textContent = stats.temporal_stability.toFixed(3);
    } catch (error) {
        console.error('Error loading statistics:', error);
    }
}

// Update feature statistics display
function updateFeatureStats(stats) {
    document.getElementById('maxAttention').textContent = 
        typeof stats.max_attention === 'number' ? stats.max_attention.toFixed(3) : stats.max_attention;
    document.getElementById('meanAttention').textContent = 
        typeof stats.mean_attention === 'number' ? stats.mean_attention.toFixed(3) : stats.mean_attention;
    if (stats.spatial_diversity !== undefined) {
        document.getElementById('spatialDiversity').textContent = 
            typeof stats.spatial_diversity === 'number' ? stats.spatial_diversity.toFixed(3) : stats.spatial_diversity;
    }
    if (stats.temporal_stability !== undefined) {
        document.getElementById('temporalStability').textContent = 
            typeof stats.temporal_stability === 'number' ? stats.temporal_stability.toFixed(3) : stats.temporal_stability;
    }
}

// Set temporal mode
function setTemporalMode(mode) {
    currentTemporalMode = mode;
    
    // Update UI
    document.querySelectorAll('.control-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Update visualization
    if (currentObservationId) {
        loadImageAndFeatures(currentObservationId);
    }
    
    // Show/hide slider
    document.getElementById('temporalSlider').style.display = 
        mode === 'temporal' ? 'block' : 'none';
}

// Update temporal frame
function updateTemporalFrame(frame) {
    if (temporalFrames && temporalFrames.length > frame) {
        const overlayImg = document.getElementById('obs-attention-img');
        if (overlayImg) overlayImg.src = temporalFrames[frame];
    }
}

// Set colormap
async function setColormap(colormap) {
    currentColormap = colormap;
    
    // Update UI
    document.querySelectorAll('#plasma-btn, #viridis-btn, #rdbu-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Handle RdBu_r → rdbu for button ID
    const btnId = colormap.toLowerCase().replace('_r', '') + '-btn';
    document.getElementById(btnId).classList.add('active');
    
    // If UMAP is active, turn it off
    if (isUMAPActive) {
        isUMAPActive = false;
        autoShowUMAP = false;
        document.getElementById('umapBtnText').textContent = 'Show UMAP RGB';
        document.getElementById('umapDescription').style.display = 'none';
        
        // Remove UMAP from visualization dropdown
        const vizSelect = document.getElementById('visualizationMethod');
        if (vizSelect && vizSelect.value === 'umap') {
            vizSelect.value = currentVisualization;
            const umapOption = vizSelect.querySelector('option[value="umap"]');
            if (umapOption) {
                umapOption.remove();
            }
        }
    }
    
    // For PCA visualization with cached data, just regenerate the overlay
    if (currentObservationId && currentVisualization.startsWith('pca') && currentPCAData) {
        console.log('🎨 Regenerating PCA overlay with new colormap (no refetch needed)');
        const startTime = performance.now();
        const overlayImg = document.getElementById('obs-attention-img');
        const overlayDataUrl = await generatePCAOverlay(currentPCAData.pca_values, currentColormap, currentAlpha);
        
        if (overlayImg && overlayDataUrl) {
            overlayImg.src = overlayDataUrl;
            const regenTime = performance.now() - startTime;
            console.log(`✅ Updated colormap without refetching data in ${regenTime.toFixed(1)}ms`);
        }
    } else if (currentObservationId) {
        // Reload features for non-PCA visualizations
        loadImageAndFeatures(currentObservationId);
    }
}

// Debounce timer for alpha updates
let alphaUpdateTimer = null;

// Update alpha
function updateAlpha(value) {
    currentAlpha = value / 100;
    document.getElementById('alphaValue').textContent = `${value}%`;
    
    console.log(`🎚️ Alpha changed to: ${currentAlpha}`);
    
    // Clear previous timer
    if (alphaUpdateTimer) {
        clearTimeout(alphaUpdateTimer);
    }
    
    // Debounce the actual update to prevent spam
    alphaUpdateTimer = setTimeout(() => {
        if (currentObservationId) {
            if (isUMAPActive && umapRGBData) {
                // Just update the alpha of existing UMAP visualization
                console.log('🌈 Updating UMAP alpha only');
                applyUMAPAlpha();
            } else {
                // Update alpha on existing overlay without reloading
                console.log('🎨 Updating overlay alpha without reloading features');
                updateOverlayAlpha();
            }
        }
    }, 300); // Increased debounce to 300ms to reduce calls
}

// Update overlay alpha without reloading features
async function updateOverlayAlpha() {
    const overlayContainer = document.getElementById('obs-attention-overlay');
    const overlayImg = document.getElementById('obs-attention-img');
    
    if (overlayContainer && overlayImg && overlayImg.src) {
        // For PCA visualization, regenerate the overlay with new alpha
        if (currentVisualization.startsWith('pca') && currentPCAData) {
            console.log('🎨 Regenerating PCA overlay with new alpha');
            const startTime = performance.now();
            const overlayDataUrl = await generatePCAOverlay(currentPCAData.pca_values, currentColormap, currentAlpha);
            if (overlayDataUrl) {
                overlayImg.src = overlayDataUrl;
                const regenTime = performance.now() - startTime;
                console.log(`⚡ PCA overlay regenerated in ${regenTime.toFixed(1)}ms`);
            }
        } else {
            // For server-generated overlays, just update the container opacity
            overlayContainer.style.opacity = currentAlpha;
        }
        console.log(`✅ Updated overlay alpha to ${currentAlpha} without API call`);
    } else {
        console.log('⚠️ No existing overlay to update alpha for');
    }
}

// Set visualization method
function setVisualizationMethod(method) {
    currentVisualization = method;
    
    // Reload features if observation is open
    if (currentObservationId) {
        loadImageAndFeatures(currentObservationId);
    }
}

// Toggle UMAP visualization
async function toggleUMAP() {
    if (!currentObservationId) {
        performanceMonitor.log('UMAP toggle skipped - no current observation');
        return;
    }
    
    performanceMonitor.start('toggleUMAP');
    performanceMonitor.log('Starting UMAP toggle', { currentObservationId, isUMAPActive });
    
    const btn = document.getElementById('umapBtn');
    const btnText = document.getElementById('umapBtnText');
    const loader = document.getElementById('umapLoader');
    const description = document.getElementById('umapDescription');
    const overlay = document.getElementById('obs-attention-overlay');
    
    if (isUMAPActive) {
        // Turn off UMAP
        isUMAPActive = false;
        autoShowUMAP = false;
        umapRGBData = null;
        btnText.textContent = 'Show UMAP RGB';
        description.style.display = 'none';
        
        // Restore previous visualization
        loadImageAndFeatures(currentObservationId);
    } else {
        // Turn on UMAP
        btnText.style.display = 'none';
        loader.style.display = 'inline';
        btn.disabled = true;
        
        try {
            const response = await fetch(`/api/features/${currentObservationId}/umap-rgb`);
            
            if (!response.ok) {
                throw new Error(`Failed to compute UMAP: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Store raw RGB values for client-side alpha blending
            umapRGBData = data.rgb_values;
            console.log('🌈 UMAP RGB data received:', {
                rawRGBType: typeof data.rgb_values,
                rawRGBLength: data.rgb_values ? data.rgb_values.length : 'N/A',
                imageDataType: typeof data.umap_rgb,
                coords3dShape: data.coords_3d ? data.coords_3d.length : 'N/A',
                shape: data.shape,
                sampleRGB: data.rgb_values ? data.rgb_values.slice(0, 9) : 'N/A'
            });
            
            // Show UMAP visualization
            isUMAPActive = true;
            autoShowUMAP = true;
            applyUMAPAlpha();
            
            btnText.textContent = 'Hide UMAP RGB';
            description.style.display = 'block';
            
            // Update visualization method dropdown to show UMAP
            const vizSelect = document.getElementById('visualizationMethod');
            if (vizSelect) {
                // Add UMAP option if not exists
                let umapOption = vizSelect.querySelector('option[value="umap"]');
                if (!umapOption) {
                    umapOption = document.createElement('option');
                    umapOption.value = 'umap';
                    umapOption.textContent = 'UMAP 3D projection from 1408D';
                    vizSelect.appendChild(umapOption);
                }
                vizSelect.value = 'umap';
            }
            
            // Update stats with UMAP info
            if (data.feature_stats) {
                updateFeatureStats({
                    max_attention: 'UMAP X',
                    mean_attention: 'UMAP Y',
                    spatial_diversity: 'UMAP Z',
                    temporal_stability: 'RGB'
                });
            }
            
        } catch (error) {
            performanceMonitor.log('ERROR in UMAP computation', { 
                error: error.message, 
                currentObservationId, 
                stack: error.stack 
            });
            alert(`Error computing UMAP: ${error.message}`);
            console.error('❌ UMAP error:', error);
        } finally {
            btnText.style.display = 'inline';
            loader.style.display = 'none';
            btn.disabled = false;
            performanceMonitor.end('toggleUMAP');
        }
    }
}

// Apply current alpha to UMAP visualization
function applyUMAPAlpha() {
    if (!umapRGBData) {
        console.warn('⚠️ applyUMAPAlpha called but no umapRGBData available');
        return;
    }
    
    console.log('🎨 Applying UMAP alpha visualization:', {
        umapRGBDataType: typeof umapRGBData,
        umapRGBDataLength: Array.isArray(umapRGBData) ? umapRGBData.length : 'N/A',
        umapRGBSample: Array.isArray(umapRGBData) ? umapRGBData.slice(0, 9) : 'N/A',
        currentAlpha: currentAlpha
    });
    
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
    
    // umapRGBData should be an array of RGB values in [0,1] range from server
    if (!Array.isArray(umapRGBData)) {
        console.error('❌ UMAP RGB data is not an array:', typeof umapRGBData);
        return;
    }
    
    // Check RGB data range - server returns values in [0,1] range
    const maxVal = Math.max(...umapRGBData);
    const minVal = Math.min(...umapRGBData);
    console.log(`🎨 UMAP RGB data range: min=${minVal.toFixed(3)}, max=${maxVal.toFixed(3)}, length=${umapRGBData.length}`);
    
    // Values should be in [0,1] range from server, scale to [0,255]
    console.log('📊 Scaling RGB values from [0,1] to [0,255]');
    
    // Fill with RGB values and current alpha
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            const srcY = Math.floor(y / scale);
            const srcX = Math.floor(x / scale);
            const srcIdx = srcY * 24 + srcX;
            
            const dstIdx = (y * size + x) * 4;
            
            // Scale RGB values from [0,1] to [0,255]
            const r = Math.floor(umapRGBData[srcIdx * 3] * 255);
            const g = Math.floor(umapRGBData[srcIdx * 3 + 1] * 255);
            const b = Math.floor(umapRGBData[srcIdx * 3 + 2] * 255);
            
            data[dstIdx] = Math.max(0, Math.min(255, r));      // R
            data[dstIdx + 1] = Math.max(0, Math.min(255, g));  // G
            data[dstIdx + 2] = Math.max(0, Math.min(255, b));  // B
            data[dstIdx + 3] = Math.floor(currentAlpha * 255); // A
        }
    }
    
    console.log(`🎨 Applied UMAP RGB visualization with scaled [0,1]→[0,255] values and alpha=${currentAlpha}`);
    
    ctx.putImageData(imageData, 0, 0);
    
    // Convert to base64 and display
    const overlayImg = document.getElementById('obs-attention-img');
    const overlayContainer = document.getElementById('obs-attention-overlay');
    if (overlayImg) {
        overlayImg.src = canvas.toDataURL('image/png');
        
        // Apply aspect ratio stretching
        if (currentImageAspectRatio > 1) {
            overlayImg.style.transform = `scaleX(${currentImageAspectRatio})`;
        } else if (currentImageAspectRatio < 1) {
            overlayImg.style.transform = `scaleY(${1 / currentImageAspectRatio})`;
        } else {
            overlayImg.style.transform = 'none';
        }
    }
    const showOverlay = document.getElementById('toggle-overlay').checked;
    if (overlayContainer) {
        overlayContainer.style.display = showOverlay ? 'block' : 'none';
        overlayContainer.style.opacity = '1';
    }
}

// Toggle overlay visibility
function toggleOverlay(show) {
    const overlayContainer = document.getElementById('obs-attention-overlay');
    if (overlayContainer) {
        overlayContainer.style.display = show ? 'block' : 'none';
    }
}

// Gallery visualization functions
let galleryCurrentVisualization = 'pca1';
let galleryIsUMAPActive = false;
let galleryCurrentObservationId = null;
// galleryVisionManager is already initialized in document ready

// Gallery vision feature control functions
function setGalleryTemporalMode(mode) {
    // Update UI
    const buttons = document.querySelectorAll('#vision-feature-panel .control-btn');
    buttons.forEach(btn => {
        if (btn.textContent.includes('Compressed') || btn.textContent.includes('Temporal')) {
            btn.classList.toggle('active', 
                (mode === 'mean' && btn.textContent.includes('Compressed')) ||
                (mode === 'temporal' && btn.textContent.includes('Temporal'))
            );
        }
    });
    
    document.getElementById('galleryTemporalSlider').style.display = 
        mode === 'temporal' ? 'block' : 'none';
    
    // Update vision manager
    if (galleryVisionManager) {
        galleryVisionManager.setTemporalMode(mode);
    }
}

function setGalleryVisualizationMethod(method) {
    galleryCurrentVisualization = method;
    if (galleryVisionManager) {
        galleryVisionManager.setVisualization(method);
    }
}

function updateGalleryTemporalFrame(frame) {
    if (galleryVisionManager) {
        galleryVisionManager.setTemporalFrame(parseInt(frame));
    }
}

// Duplicate removed - using the version above

// Duplicate removed - using the version above

// Duplicate removed - using the version above

function setGalleryColormap(colormap) {
    // Update UI using button IDs for reliable selection
    document.getElementById('gallery-plasma-btn')?.classList.remove('active');
    document.getElementById('gallery-viridis-btn')?.classList.remove('active');
    document.getElementById('gallery-rdbu-btn')?.classList.remove('active');
    
    // Add active class to the selected button
    const btnIdMap = {
        'plasma': 'gallery-plasma-btn',
        'viridis': 'gallery-viridis-btn',
        'RdBu_r': 'gallery-rdbu-btn'
    };
    
    const btnId = btnIdMap[colormap];
    if (btnId) {
        document.getElementById(btnId)?.classList.add('active');
    }
    
    // Update vision manager with normalized colormap name
    if (galleryVisionManager) {
        galleryVisionManager.setColormap(colormap);
    }
    
    // Update the old gallery visualization system if still in use
    galleryColormap = colormap;
    if (typeof updateGalleryVisualization === 'function') {
        updateGalleryVisualization();
    }
}

function updateGalleryAlpha(value) {
    document.getElementById('galleryAlphaValue').textContent = value + '%';
    
    // Update vision manager
    if (galleryVisionManager) {
        galleryVisionManager.setAlpha(value / 100);
    }
    
    // Update the old gallery visualization system if still in use
    galleryAlpha = value / 100;
    if (typeof updateGalleryVisualization === 'function' && window.currentGalleryImageId) {
        updateGalleryVisualization();
    }
}

async function toggleGalleryUMAP() {
    if (!galleryVisionManager) return;
    
    const btnText = document.getElementById('galleryUmapBtnText');
    const loader = document.getElementById('galleryUmapLoader');
    const description = document.getElementById('galleryUmapDescription');
    
    try {
        // Show loader, hide text
        btnText.style.display = 'none';
        loader.style.display = 'inline';
        
        const wasActive = await galleryVisionManager.toggleUMAP();
        
        // Hide loader, show text
        loader.style.display = 'none';
        btnText.style.display = 'inline';
        btnText.textContent = wasActive ? 'Hide UMAP RGB' : 'Show UMAP RGB';
        
        // Show/hide description
        description.style.display = wasActive ? 'block' : 'none';
        
        galleryIsUMAPActive = wasActive;
    } catch (error) {
        console.error('Error toggling UMAP:', error);
        loader.style.display = 'none';
        btnText.style.display = 'inline';
        btnText.textContent = 'Show UMAP RGB';
    }
}

function updateGalleryStats(stats) {
    // Update gallery statistics display
    if (stats.max_attention !== undefined) {
        document.getElementById('galleryMaxAttention').textContent = stats.max_attention;
    }
    if (stats.mean_attention !== undefined) {
        document.getElementById('galleryMeanAttention').textContent = stats.mean_attention;
    }
    if (stats.spatial_diversity !== undefined) {
        document.getElementById('gallerySpatialDiversity').textContent = stats.spatial_diversity;
    }
    if (stats.temporal_stability !== undefined) {
        document.getElementById('galleryTemporalStability').textContent = stats.temporal_stability;
    }
}

function setGalleryTemporalMode(mode) {
    if (!galleryVisionManager) return;
    
    // Update button states
    document.querySelectorAll('#vision-feature-panel .control-buttons button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    // Show/hide temporal slider
    const slider = document.getElementById('galleryTemporalSlider');
    slider.style.display = mode === 'temporal' ? 'block' : 'none';
    
    // Update vision manager
    galleryVisionManager.setTemporalMode(mode);
}

function updateGalleryTemporalFrame(value) {
    if (!galleryVisionManager) return;
    galleryVisionManager.setTemporalFrame(parseInt(value));
}

function setGalleryVisualizationMethod(method) {
    if (!galleryVisionManager) return;
    galleryVisionManager.setVisualization(method);
}

// Gallery UMAP visualization is now handled by galleryVisionManager

// Gallery visualization is now handled by galleryVisionManager

function toggleGalleryOverlay(show) {
    if (galleryVisionManager) {
        galleryVisionManager.toggleOverlay(show);
    }
}

// Animate camera to focus on object
function flyToObject(object) {
    if (!object || !window.controls3D) return;
    
    const startPosition = camera3D.position.clone();
    const startTarget = window.controls3D.target.clone();
    
    // Calculate desired screen fraction and corresponding distance
    const sphereRadius = object.geometry.parameters.radius;
    const desiredScreenFraction = 0.05;
    const distance = (sphereRadius * 2) / (Math.tan((camera3D.fov * Math.PI / 180) / 2) * desiredScreenFraction);
    
    // Calculate new position and target
    const objectPosition = object.position.clone();
    
    // Position camera at a nice angle
    const endPosition = objectPosition.clone();
    endPosition.x += distance * 0.5;
    endPosition.y += distance * 0.3;
    endPosition.z += distance * 0.5;
    
    const endTarget = objectPosition.clone();
    
    // Animate
    const duration = 1500; // 1.5 seconds like original
    const startTime = Date.now();
    
    function animate() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        // Easing function - same as original
        const eased = 1 - Math.pow(1 - progress, 3);
        
        // Interpolate position
        camera3D.position.lerpVectors(startPosition, endPosition, eased);
        
        // Interpolate target
        window.controls3D.target.lerpVectors(startTarget, endTarget, eased);
        window.controls3D.update();
        
        // Look at target
        camera3D.lookAt(window.controls3D.target);
        
        if (progress < 1) {
            requestAnimationFrame(animate);
        }
    }
    
    animate();
}