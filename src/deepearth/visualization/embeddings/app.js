/**
 * 3D Embeddings Visualization
 * Production-ready client for visualizing high-dimensional embeddings
 */

class EmbeddingVisualization {
    constructor(config = {}) {
        this.config = config;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.data = null;
        this.points = [];
        this.sphereGroup = null;
        this.textGroup = null;
        
        this.selectedObject = null;
        this.hoveredObject = null;
        
        // Settings with defaults
        this.settings = {
            sphereSize: config.visualization?.defaultSphereSize || 0.05,
            textSize: config.visualization?.defaultTextSize || 0.3,
            showLabels: true,
            autoRotate: false,
            debugMode: false
        };
        
        // Data field mappings
        this.fields = config.data?.fields || {
            id: 'id',
            label: 'name',
            cluster: 'cluster',
            metadata: []
        };
    }
    
    async initialize() {
        try {
            console.log('Starting initialization...');
            
            // Load configuration if not provided
            if (!this.config.visualization) {
                console.log('Loading configuration from server...');
                await this.loadConfig();
            }
            console.log('Configuration:', this.config);
            
            // Load data
            console.log('Loading data...');
            await this.loadData();
            console.log(`Data loaded: ${this.data.points.length} points`);
            
            // Setup visualization
            console.log('Setting up scene...');
            this.setupScene();
            
            console.log('Setting up lighting...');
            this.setupLighting();
            
            console.log('Creating visualization objects...');
            this.createVisualization();
            
            console.log('Setting up interactions...');
            this.setupInteraction();
            
            console.log('Setting up controls...');
            this.setupControls();
            
            // Hide loading, show UI
            document.getElementById('loading').style.display = 'none';
            document.getElementById('info-panel').style.display = 'block';
            document.getElementById('controls').style.display = 'block';
            
            // Show cluster legend if we have clusters
            if (this.data.stats?.cluster_sizes || this.data.colors) {
                document.getElementById('cluster-legend').style.display = 'block';
            }
            
            // Update statistics
            this.updateStats();
            
            // Start animation loop
            this.animate();
            
            console.log('Visualization initialized successfully');
            
        } catch (error) {
            console.error('Failed to initialize visualization:', error);
            console.error('Stack trace:', error.stack);
            this.showError('Failed to load visualization. Please check the console.');
        }
    }
    
    async loadConfig() {
        try {
            const response = await fetch('/config');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            this.config = await response.json();
            this.fields = this.config.data?.fields || this.fields;
        } catch (error) {
            console.warn('Failed to load config, using defaults:', error);
        }
    }
    
    async loadData() {
        console.log('Fetching data from /data endpoint...');
        const response = await fetch('/data');
        console.log('Data response status:', response.status);
        
        if (!response.ok) {
            const text = await response.text();
            console.error('Failed to load data:', text);
            throw new Error(`Failed to load data: HTTP ${response.status}`);
        }
        
        this.data = await response.json();
        console.log('Raw data:', this.data);
        
        // Validate data structure
        if (!this.data.points || !Array.isArray(this.data.points)) {
            console.error('Invalid data structure:', this.data);
            throw new Error('Invalid data format: missing points array');
        }
        
        // Apply scale factor to coordinates
        const scaleFactor = this.config.visualization?.scaleFactor || 3.0;
        console.log(`Applying scale factor: ${scaleFactor}`);
        
        this.data.points.forEach(point => {
            point.x = (point.x || 0) * scaleFactor;
            point.y = (point.y || 0) * scaleFactor;
            point.z = (point.z || 0) * scaleFactor;
        });
        
        console.log(`Loaded ${this.data.points.length} data points`);
        console.log('Sample point:', this.data.points[0]);
    }
    
    setupScene() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0a0a);
        this.scene.fog = new THREE.Fog(0x0a0a0a, 100, 300);
        
        // Camera
        const aspect = window.innerWidth / window.innerHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.1, 1000);
        
        // Calculate optimal camera position
        const bounds = this.calculateBounds();
        const center = this.calculateCenter();
        const maxDimension = Math.max(
            bounds.max.x - bounds.min.x,
            bounds.max.y - bounds.min.y,
            bounds.max.z - bounds.min.z
        );
        
        // Position camera
        const distance = maxDimension * 1.5;
        this.camera.position.set(
            center.x + distance * 0.5,
            center.y + distance * 0.3,
            center.z + distance * 0.5
        );
        
        // Renderer
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true
        });
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        
        document.getElementById('canvas-container').appendChild(this.renderer.domElement);
        
        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.target.copy(center);
        this.camera.lookAt(center);
    }
    
    setupLighting() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
        this.scene.add(ambientLight);
        
        // Directional light
        const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
        dirLight.position.set(10, 20, 10);
        dirLight.castShadow = true;
        this.scene.add(dirLight);
        
        // Accent lights
        const accentLight1 = new THREE.DirectionalLight(0x4fc3f7, 0.2);
        accentLight1.position.set(-20, 10, -20);
        this.scene.add(accentLight1);
        
        const accentLight2 = new THREE.DirectionalLight(0xff6b6b, 0.2);
        accentLight2.position.set(20, 10, 20);
        this.scene.add(accentLight2);
    }
    
    createVisualization() {
        this.sphereGroup = new THREE.Group();
        this.textGroup = new THREE.Group();
        
        const sphereGeometry = new THREE.SphereGeometry(this.settings.sphereSize, 24, 24);
        
        // Create materials for clusters
        const materials = {};
        const colors = this.data.colors || {};
        
        Object.keys(colors).forEach(clusterId => {
            const color = new THREE.Color(colors[clusterId]);
            materials[clusterId] = new THREE.MeshPhysicalMaterial({
                color: color,
                metalness: 0.1,
                roughness: 0.4,
                clearcoat: 0.3,
                clearcoatRoughness: 0.4,
                emissive: color,
                emissiveIntensity: 0.05
            });
        });
        
        // Default material for points without cluster
        const defaultMaterial = new THREE.MeshPhysicalMaterial({
            color: 0x888888,
            metalness: 0.1,
            roughness: 0.4
        });
        
        // Create spheres and labels
        this.data.points.forEach(point => {
            // Get cluster value
            const cluster = point[this.fields.cluster];
            const material = materials[cluster] || defaultMaterial;
            
            // Create sphere
            const sphere = new THREE.Mesh(sphereGeometry, material);
            sphere.position.set(point.x, point.y, point.z);
            sphere.castShadow = true;
            sphere.receiveShadow = true;
            sphere.userData = point;
            this.sphereGroup.add(sphere);
            
            // Create label
            const label = point[this.fields.label] || `Point ${point[this.fields.id]}`;
            const sprite = this.createTextSprite(label);
            sprite.position.set(point.x, point.y + this.settings.sphereSize * 3, point.z);
            sprite.userData = point;
            this.textGroup.add(sprite);
        });
        
        this.scene.add(this.sphereGroup);
        this.scene.add(this.textGroup);
        
        // Create cluster legend if we have cluster data
        if (this.data.stats?.cluster_sizes) {
            this.createClusterLegend();
        }
    }
    
    createTextSprite(text) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        
        // Configure font
        const fontSize = 32;
        context.font = `${fontSize}px 'Inter', sans-serif`;
        
        // Measure text
        const metrics = context.measureText(text);
        const textWidth = metrics.width;
        
        // Set canvas size
        canvas.width = textWidth + 20;
        canvas.height = fontSize + 10;
        
        // Draw text
        context.font = `${fontSize}px 'Inter', sans-serif`;
        context.fillStyle = 'white';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.shadowColor = 'rgba(0, 0, 0, 0.8)';
        context.shadowBlur = 8;
        context.fillText(text, canvas.width / 2, canvas.height / 2);
        
        // Create sprite
        const texture = new THREE.CanvasTexture(canvas);
        texture.minFilter = THREE.LinearFilter;
        
        const spriteMaterial = new THREE.SpriteMaterial({ 
            map: texture,
            transparent: true,
            opacity: 0.9
        });
        
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.scale.set(
            canvas.width / 150 * this.settings.textSize, 
            canvas.height / 150 * this.settings.textSize, 
            1
        );
        
        return sprite;
    }
    
    setupInteraction() {
        // Mouse events
        this.renderer.domElement.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.renderer.domElement.addEventListener('click', (e) => this.onClick(e));
        window.addEventListener('resize', () => this.onWindowResize());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.clearSelection();
            } else if (e.key === 'ArrowLeft' && this.currentImages && this.currentImages.length > 1) {
                this.navigateImages(-1);
            } else if (e.key === 'ArrowRight' && this.currentImages && this.currentImages.length > 1) {
                this.navigateImages(1);
            }
        });
    }
    
    setupControls() {
        // Get control elements
        const controls = {
            sphereSize: document.getElementById('sphere-size'),
            textSize: document.getElementById('text-size'),
            showLabels: document.getElementById('show-labels'),
            autoRotate: document.getElementById('auto-rotate'),
            debugMode: document.getElementById('debug-mode'),
            searchBox: document.getElementById('search-box')
        };
        
        // Sphere size
        if (controls.sphereSize) {
            controls.sphereSize.value = this.settings.sphereSize;
            controls.sphereSize.addEventListener('input', (e) => {
                this.settings.sphereSize = parseFloat(e.target.value);
                this.updateSphereSize();
            });
        }
        
        // Text size
        if (controls.textSize) {
            controls.textSize.value = this.settings.textSize;
            controls.textSize.addEventListener('input', (e) => {
                this.settings.textSize = parseFloat(e.target.value);
                this.updateTextSize();
            });
        }
        
        // Show labels
        if (controls.showLabels) {
            controls.showLabels.checked = this.settings.showLabels;
            controls.showLabels.addEventListener('change', (e) => {
                this.settings.showLabels = e.target.checked;
                this.textGroup.visible = this.settings.showLabels;
            });
        }
        
        // Auto rotate
        if (controls.autoRotate) {
            controls.autoRotate.addEventListener('change', (e) => {
                this.settings.autoRotate = e.target.checked;
                this.controls.autoRotate = this.settings.autoRotate;
                this.controls.autoRotateSpeed = 0.5;
            });
        }
        
        // Search
        if (controls.searchBox) {
            controls.searchBox.addEventListener('input', (e) => {
                this.highlightSearchResults(e.target.value.toLowerCase());
            });
        }
    }
    
    updateStats() {
        const stats = this.data.stats || {};
        
        // Update stat displays
        const elements = {
            'total-points': stats.total_points || this.data.points.length,
            'total-clusters': stats.n_clusters || Object.keys(this.data.colors || {}).length,
            'data-dimensions': stats.dimensions || 'Unknown'
        };
        
        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }
    
    createClusterLegend() {
        const container = document.getElementById('cluster-list');
        if (!container) return;
        
        container.innerHTML = '';
        const clusterSizes = this.data.stats.cluster_sizes || {};
        
        Object.entries(clusterSizes).forEach(([clusterId, size]) => {
            const item = document.createElement('div');
            item.className = 'cluster-item';
            
            const color = this.data.colors?.[clusterId] || '#888888';
            item.innerHTML = `
                <div class="cluster-color" style="background-color: ${color}"></div>
                <div>Cluster ${clusterId} (${size} points)</div>
            `;
            
            item.addEventListener('click', () => this.focusOnCluster(parseInt(clusterId)));
            container.appendChild(item);
        });
    }
    
    calculateBounds() {
        const bounds = {
            min: { x: Infinity, y: Infinity, z: Infinity },
            max: { x: -Infinity, y: -Infinity, z: -Infinity }
        };
        
        this.data.points.forEach(point => {
            bounds.min.x = Math.min(bounds.min.x, point.x);
            bounds.min.y = Math.min(bounds.min.y, point.y);
            bounds.min.z = Math.min(bounds.min.z, point.z);
            bounds.max.x = Math.max(bounds.max.x, point.x);
            bounds.max.y = Math.max(bounds.max.y, point.y);
            bounds.max.z = Math.max(bounds.max.z, point.z);
        });
        
        return bounds;
    }
    
    calculateCenter() {
        const center = new THREE.Vector3();
        
        this.data.points.forEach(point => {
            center.x += point.x;
            center.y += point.y;
            center.z += point.z;
        });
        
        center.divideScalar(this.data.points.length);
        return center;
    }
    
    onMouseMove(event) {
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const sphereIntersects = this.raycaster.intersectObjects(this.sphereGroup.children);
        const textIntersects = this.raycaster.intersectObjects(this.textGroup.children);
        
        const intersects = [...sphereIntersects, ...textIntersects]
            .sort((a, b) => a.distance - b.distance);
        
        if (intersects.length > 0) {
            const object = intersects[0].object;
            if (this.hoveredObject !== object) {
                this.clearHover();
                this.hoveredObject = object;
                if (object.type === 'Mesh') {
                    object.scale.set(1.5, 1.5, 1.5);
                }
                document.body.style.cursor = 'pointer';
            }
        } else {
            this.clearHover();
        }
    }
    
    clearHover() {
        if (this.hoveredObject) {
            if (this.hoveredObject.type === 'Mesh') {
                this.hoveredObject.scale.set(1, 1, 1);
            }
            this.hoveredObject = null;
            document.body.style.cursor = 'default';
        }
    }
    
    onClick(event) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        
        const sphereIntersects = this.raycaster.intersectObjects(this.sphereGroup.children);
        const textIntersects = this.raycaster.intersectObjects(this.textGroup.children);
        
        const intersects = [...sphereIntersects, ...textIntersects]
            .sort((a, b) => a.distance - b.distance);
        
        if (intersects.length > 0) {
            const object = intersects[0].object;
            this.selectPoint(object.userData);
            
            // Find corresponding sphere for camera animation
            const sphere = this.sphereGroup.children.find(
                s => s.userData[this.fields.id] === object.userData[this.fields.id]
            );
            if (sphere) {
                this.flyToObject(sphere);
            }
        }
    }
    
    selectPoint(data) {
        this.selectedObject = data;
        
        // Update info panel
        const infoDiv = document.getElementById('point-info');
        if (!infoDiv) return;
        
        infoDiv.style.display = 'block';
        
        // Update name
        const nameElement = document.getElementById('point-name');
        if (nameElement) {
            nameElement.textContent = data[this.fields.label] || `Point ${data[this.fields.id]}`;
        }
        
        // Update cluster info
        const clusterElement = document.getElementById('point-cluster');
        if (clusterElement) {
            const cluster = data[this.fields.cluster];
            const color = this.data.colors?.[cluster] || '#888888';
            clusterElement.innerHTML = `
                <span class="cluster-info" style="background-color: ${color}; 
                      padding: 4px 8px; border-radius: 4px;">
                    Cluster ${cluster}
                </span>
            `;
        }
        
        // Update metadata
        const metadataElement = document.getElementById('point-metadata');
        if (metadataElement && this.fields.metadata.length > 0) {
            metadataElement.style.display = 'block';
            metadataElement.innerHTML = '<h4>Additional Information:</h4>';
            
            this.fields.metadata.forEach(field => {
                if (data[field] !== undefined) {
                    const div = document.createElement('div');
                    div.className = 'metadata-item';
                    div.innerHTML = `<strong>${field}:</strong> ${data[field]}`;
                    metadataElement.appendChild(div);
                }
            });
        }
        
        // Load images if enabled
        if (this.config.images?.enabled) {
            this.loadPointImages(data);
        }
    }
    
    async loadPointImages(data) {
        // Create or get image display container
        let imageContainer = document.getElementById('point-images');
        if (!imageContainer) {
            const pointInfo = document.getElementById('point-info');
            imageContainer = document.createElement('div');
            imageContainer.id = 'point-images';
            pointInfo.appendChild(imageContainer);
        }
        
        imageContainer.innerHTML = '<div class="loading-image">Loading images...</div>';
        
        try {
            const imgConfig = this.config.images;
            
            // Load image summary if not already loaded
            if (!this.imageSummary && imgConfig.summaryFile) {
                const response = await fetch('/image_summary');
                if (response.ok) {
                    this.imageSummary = await response.json();
                    console.log('Loaded image summary:', Object.keys(this.imageSummary).length, 'entries');
                }
            }
            
            // Get the key to look up in summary
            const summaryKey = data[imgConfig.summaryKeyField || this.fields.label];
            console.log('Looking for images with key:', summaryKey);
            
            let imageDirectory = '';
            let imagePattern = imgConfig.filePattern || 'img_{index}.jpg';
            let maxImages = imgConfig.maxImages || 10;
            
            // If we have a summary file, use it to get image info
            if (this.imageSummary && this.imageSummary[summaryKey]) {
                const imageInfo = this.imageSummary[summaryKey];
                console.log('Found image info:', imageInfo);
                
                // Get the image identifier from summary
                const imageId = imageInfo[imgConfig.summaryImageField || 'id'];
                
                // Build directory name using pattern
                if (imgConfig.directoryPattern) {
                    imageDirectory = this.replacePlaceholders(imgConfig.directoryPattern, {
                        ...data,
                        ...imageInfo,
                        [imgConfig.summaryImageField]: imageId
                    });
                } else {
                    imageDirectory = imageId;
                }
                
                // Update max images if specified in summary
                if (imageInfo.image_count) {
                    maxImages = Math.min(maxImages, imageInfo.image_count);
                }
            } else if (!imgConfig.summaryFile) {
                // No summary file - build directory from data directly
                if (imgConfig.directoryPattern) {
                    imageDirectory = this.replacePlaceholders(imgConfig.directoryPattern, data);
                } else {
                    imageDirectory = data[imgConfig.imageIdField || this.fields.id];
                }
            } else {
                imageContainer.innerHTML = '<div class="no-images">No images available</div>';
                return;
            }
            
            // Construct base URL
            const baseUrl = this.replacePlaceholders(imgConfig.urlPattern, {
                directory: imageDirectory,
                ...data
            });
            
            console.log('Attempting to load images from:', baseUrl);
            
            // Try to load directory listing
            const response = await fetch(baseUrl);
            let images = [];
            
            if (response.ok) {
                const dirData = await response.json();
                images = dirData.files ? dirData.files.filter(f => f.match(/\.(jpg|jpeg|png|gif|webp)$/i)) : [];
                console.log('Found', images.length, 'images in directory');
            }
            
            // If no images found via directory listing, try pattern-based approach
            if (images.length === 0 && imagePattern.includes('{index}')) {
                for (let i = 1; i <= maxImages; i++) {
                    const filename = imagePattern.replace('{index}', i);
                    images.push(filename);
                }
                // We'll rely on onerror to handle missing images
            }
            
            if (images.length === 0) {
                imageContainer.innerHTML = '<div class="no-images">No images found</div>';
                return;
            }
            
            // Display images
            this.currentImages = images.slice(0, maxImages).map(img => baseUrl + img);
            this.currentImageIndex = 0;
            
            const itemLabel = data[this.fields.label] || 'Item';
            
            imageContainer.innerHTML = `
                <div class="image-gallery-container">
                    <img class="gallery-image" 
                         src="${this.currentImages[0]}" 
                         alt="${itemLabel}"
                         onerror="this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'300\\' height=\\'200\\'%3E%3Crect width=\\'300\\' height=\\'200\\' fill=\\'%23333\\'/%3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' text-anchor=\\'middle\\' dy=\\'.3em\\' fill=\\'%23666\\' font-family=\\'sans-serif\\' font-size=\\'14\\'%3EImage not available%3C/text%3E%3C/svg%3E'">
                    ${this.currentImages.length > 1 ? `
                        <div class="image-nav image-prev" onclick="window.visualization.navigateImages(-1)">‹</div>
                        <div class="image-nav image-next" onclick="window.visualization.navigateImages(1)">›</div>
                        <div class="image-counter">${this.currentImageIndex + 1} / ${this.currentImages.length}</div>
                    ` : ''}
                </div>
            `;
        } catch (error) {
            console.error('Error loading images:', error);
            imageContainer.innerHTML = '<div class="no-images">Error loading images</div>';
        }
    }
    
    replacePlaceholders(template, data) {
        return template.replace(/\{(\w+)\}/g, (match, key) => {
            return data[key] !== undefined ? data[key] : match;
        });
    }
    
    navigateImages(direction) {
        if (!this.currentImages || this.currentImages.length <= 1) return;
        
        this.currentImageIndex = (this.currentImageIndex + direction + this.currentImages.length) % this.currentImages.length;
        
        const img = document.querySelector('.gallery-image');
        const counter = document.querySelector('.image-counter');
        
        if (img) {
            img.src = this.currentImages[this.currentImageIndex];
        }
        if (counter) {
            counter.textContent = `${this.currentImageIndex + 1} / ${this.currentImages.length}`;
        }
    }
    
    clearSelection() {
        this.selectedObject = null;
        const infoDiv = document.getElementById('point-info');
        if (infoDiv) {
            infoDiv.style.display = 'none';
        }
    }
    
    flyToObject(object) {
        const targetPosition = object.position.clone();
        
        // Calculate appropriate distance based on sphere size
        // Use a much larger multiplier to avoid zooming too close
        const sphereDiameter = this.settings.sphereSize * 2;
        const fov = this.camera.fov * Math.PI / 180;
        // Increased from 0.25 to 0.05 to zoom out more
        const desiredScreenFraction = 0.05;
        const distance = (sphereDiameter / desiredScreenFraction) / (2 * Math.tan(fov / 2));
        
        // Calculate new camera position
        const direction = this.camera.position.clone().sub(targetPosition).normalize();
        const newCameraPosition = targetPosition.clone().add(direction.multiplyScalar(distance));
        
        // Animate camera
        const startPosition = this.camera.position.clone();
        const startTarget = this.controls.target.clone();
        
        const duration = 1500;
        const startTime = Date.now();
        
        const animateCamera = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            // Easing function
            const eased = 1 - Math.pow(1 - progress, 3);
            
            this.camera.position.lerpVectors(startPosition, newCameraPosition, eased);
            this.controls.target.lerpVectors(startTarget, targetPosition, eased);
            
            if (progress < 1) {
                requestAnimationFrame(animateCamera);
            }
        };
        
        animateCamera();
    }
    
    focusOnCluster(clusterId) {
        const clusterPoints = this.data.points.filter(
            p => p[this.fields.cluster] === clusterId
        );
        
        if (clusterPoints.length === 0) return;
        
        // Calculate cluster center
        const center = new THREE.Vector3();
        clusterPoints.forEach(point => {
            center.x += point.x;
            center.y += point.y;
            center.z += point.z;
        });
        center.divideScalar(clusterPoints.length);
        
        // Calculate bounds
        let maxDistance = 0;
        clusterPoints.forEach(point => {
            const distance = Math.sqrt(
                Math.pow(point.x - center.x, 2) +
                Math.pow(point.y - center.y, 2) +
                Math.pow(point.z - center.z, 2)
            );
            maxDistance = Math.max(maxDistance, distance);
        });
        
        // Position camera to see entire cluster
        const cameraDistance = maxDistance * 3;
        const direction = this.camera.position.clone().sub(center).normalize();
        const newPosition = center.clone().add(direction.multiplyScalar(cameraDistance));
        
        // Animate to new position
        this.animateCameraToPosition(newPosition, center);
    }
    
    animateCameraToPosition(position, target) {
        const startPosition = this.camera.position.clone();
        const startTarget = this.controls.target.clone();
        
        const duration = 2000;
        const startTime = Date.now();
        
        const animate = () => {
            const elapsed = Date.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const eased = 1 - Math.pow(1 - progress, 3);
            
            this.camera.position.lerpVectors(startPosition, position, eased);
            this.controls.target.lerpVectors(startTarget, target, eased);
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        animate();
    }
    
    highlightSearchResults(query) {
        if (!query) {
            // Reset all opacities
            this.sphereGroup.children.forEach(sphere => {
                sphere.material.opacity = 1;
                sphere.material.transparent = false;
            });
            this.textGroup.children.forEach(text => {
                text.material.opacity = 0.9;
            });
            return;
        }
        
        // Highlight matching points
        this.sphereGroup.children.forEach((sphere, index) => {
            const label = sphere.userData[this.fields.label] || '';
            const matches = label.toLowerCase().includes(query);
            
            sphere.material.opacity = matches ? 1 : 0.1;
            sphere.material.transparent = true;
            
            if (this.textGroup.children[index]) {
                this.textGroup.children[index].material.opacity = matches ? 0.9 : 0.1;
            }
        });
    }
    
    updateSphereSize() {
        const newGeometry = new THREE.SphereGeometry(this.settings.sphereSize, 24, 24);
        this.sphereGroup.children.forEach(sphere => {
            sphere.geometry.dispose();
            sphere.geometry = newGeometry;
        });
    }
    
    updateTextSize() {
        this.textGroup.children.forEach((sprite, index) => {
            const point = this.data.points[index];
            const label = point[this.fields.label] || `Point ${point[this.fields.id]}`;
            
            // Recreate sprite with new size
            sprite.material.map.dispose();
            const newSprite = this.createTextSprite(label);
            sprite.material.map = newSprite.material.map;
            sprite.scale.copy(newSprite.scale);
        });
    }
    
    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }
    
    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }
    
    showError(message) {
        const loadingDiv = document.getElementById('loading');
        if (loadingDiv) {
            loadingDiv.innerHTML = `
                <div style="color: #ff4444; text-align: center; padding: 20px;">
                    <div style="font-size: 1.2em; margin-bottom: 10px;">Error</div>
                    <div>${message}</div>
                </div>
            `;
        }
    }
}

// Initialize visualization when page loads
window.addEventListener('load', async () => {
    console.log('Page loaded, initializing visualization...');
    
    // Load configuration from server
    try {
        console.log('Fetching config from /config...');
        const response = await fetch('/config');
        console.log('Config response:', response.status);
        
        const config = response.ok ? await response.json() : {};
        console.log('Loaded config:', config);
        
        // Create and initialize visualization
        window.visualization = new EmbeddingVisualization(config);
        await window.visualization.initialize();
        
    } catch (error) {
        console.error('Failed to initialize:', error);
        console.error('Error details:', error.message);
        console.error('Stack:', error.stack);
        
        const loadingDiv = document.getElementById('loading');
        if (loadingDiv) {
            loadingDiv.innerHTML = `
                <div style="color: #ff4444; text-align: center; padding: 20px;">
                    <div style="font-size: 1.2em; margin-bottom: 10px;">Error</div>
                    <div>${error.message}</div>
                    <div style="font-size: 0.9em; margin-top: 10px;">Check browser console for details</div>
                </div>
            `;
        }
    }
});