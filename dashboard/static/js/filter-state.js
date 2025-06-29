// Global Filter State Management for DeepEarth Dashboard
// This module manages filters that persist across Geospatial and Embeddings views

class FilterStateManager {
    constructor() {
        // Initialize filter state with defaults
        // These will be updated with actual data ranges when data loads
        this.defaultState = {
            temporal: {
                yearMin: null, // Will be set from data
                yearMax: null, // Will be set from data
                monthMin: 1,
                monthMax: 12,
                hourMin: 0,
                hourMax: 23
            },
            geographic: null, // { north, south, east, west } or null for all data
            species: null, // Selected species taxon_id or null for all
            gridCell: null // Selected grid cell bounds or null
        };
        
        // Deep copy default state
        this.state = JSON.parse(JSON.stringify(this.defaultState));
        
        // Callbacks for filter changes
        this.listeners = [];
        
        // Do NOT load from localStorage - keep filters session-specific
        // this.loadState(); // REMOVED to prevent cross-session/cross-user persistence
        
        // Clean up any existing localStorage data
        if (typeof localStorage !== 'undefined') {
            localStorage.removeItem('deepearth_filters');
        }
    }
    
    // State management - NO LONGER PERSISTS TO DISK
    saveState() {
        // Do nothing - we don't want to persist across sessions
        // This method is kept for compatibility but does not save to localStorage
    }
    
    loadState() {
        // Do nothing - we don't want to load from previous sessions
        // This method is kept for compatibility but does not load from localStorage
    }
    
    // Initialize temporal filters from actual data range
    initializeFromDataRange(minYear, maxYear) {
        if (this.defaultState.temporal.yearMin === null) {
            this.defaultState.temporal.yearMin = minYear;
            this.defaultState.temporal.yearMax = maxYear;
            this.state.temporal.yearMin = minYear;
            this.state.temporal.yearMax = maxYear;
            
            console.log(`Initialized temporal filters from data: ${minYear} - ${maxYear}`);
            
            // Apply to UI
            this.applyToUI();
        }
    }
    
    // Filter setters
    setTemporalFilter(type, value) {
        this.state.temporal[type] = value;
        this.saveState();
        this.notifyListeners('temporal', this.state.temporal);
    }
    
    setGeographicFilter(bounds) {
        this.state.geographic = bounds;
        this.saveState();
        this.notifyListeners('geographic', bounds);
    }
    
    setSpeciesFilter(taxonId) {
        this.state.species = taxonId;
        this.saveState();
        this.notifyListeners('species', taxonId);
    }
    
    setGridCellFilter(bounds) {
        this.state.gridCell = bounds;
        // Grid cell is a type of geographic filter
        if (bounds) {
            this.setGeographicFilter(bounds);
        }
    }
    
    clearGeographicFilter() {
        this.state.geographic = null;
        this.state.gridCell = null;
        this.saveState();
        this.notifyListeners('geographic', null);
    }
    
    // Get current filters
    getFilters() {
        return {
            ...this.state.temporal,
            geographic: this.state.geographic,
            species: this.state.species,
            gridCell: this.state.gridCell
        };
    }
    
    // Get filter description for UI
    getFilterDescription() {
        const parts = [];
        
        // Temporal filter
        if (this.defaultState.temporal.yearMin !== null && 
            (this.state.temporal.yearMin !== this.defaultState.temporal.yearMin || 
             this.state.temporal.yearMax !== this.defaultState.temporal.yearMax)) {
            parts.push(`Years: ${this.state.temporal.yearMin}-${this.state.temporal.yearMax}`);
        }
        
        if (this.state.temporal.monthMin !== 1 || this.state.temporal.monthMax !== 12) {
            parts.push(`Months: ${this.state.temporal.monthMin}-${this.state.temporal.monthMax}`);
        }
        
        if (this.state.temporal.hourMin !== 0 || this.state.temporal.hourMax !== 23) {
            parts.push(`Hours: ${this.state.temporal.hourMin}-${this.state.temporal.hourMax}`);
        }
        
        // Geographic filter
        if (this.state.gridCell) {
            parts.push('Grid Cell Selected');
        } else if (this.state.geographic) {
            parts.push('Geographic Bounds Set');
        }
        
        // Species filter
        if (this.state.species) {
            parts.push('Species Filtered');
        }
        
        return parts.length > 0 ? parts.join(' • ') : 'All Data';
    }
    
    // Apply filters to UI elements
    applyToUI() {
        // Apply temporal filters to both sets of controls
        const controls = [
            { prefix: '', view: 'geospatial' },
            { prefix: 'emb-', view: 'embeddings' }
        ];
        
        controls.forEach(({ prefix }) => {
            const yearMin = document.getElementById(`${prefix}year-min`);
            const yearMax = document.getElementById(`${prefix}year-max`);
            const monthMin = document.getElementById(`${prefix}month-min`);
            const monthMax = document.getElementById(`${prefix}month-max`);
            const hourMin = document.getElementById(`${prefix}hour-min`);
            const hourMax = document.getElementById(`${prefix}hour-max`);
            
            if (yearMin) yearMin.value = this.state.temporal.yearMin;
            if (yearMax) yearMax.value = this.state.temporal.yearMax;
            if (monthMin) monthMin.value = this.state.temporal.monthMin;
            if (monthMax) monthMax.value = this.state.temporal.monthMax;
            if (hourMin) hourMin.value = this.state.temporal.hourMin;
            if (hourMax) hourMax.value = this.state.temporal.hourMax;
        });
        
        // Update filter display
        const filterValue = document.getElementById('filter-value');
        if (filterValue) {
            filterValue.textContent = this.getFilterDescription();
        }
    }
    
    // Event listeners
    addListener(callback) {
        this.listeners.push(callback);
    }
    
    removeListener(callback) {
        this.listeners = this.listeners.filter(l => l !== callback);
    }
    
    notifyListeners(type, data) {
        this.listeners.forEach(callback => {
            try {
                callback(type, data, this.getFilters());
            } catch (e) {
                console.error('Error in filter listener:', e);
            }
        });
    }
    
    // Check if filters are at default values
    hasNonDefaultFilters() {
        // Check temporal filters
        const temporal = this.state.temporal;
        const defaultTemporal = this.defaultState.temporal;
        
        // If defaults haven't been initialized yet, no filters are active
        if (defaultTemporal.yearMin === null || defaultTemporal.yearMax === null) {
            return false;
        }
        
        const hasTemporalFilter = (
            temporal.yearMin !== defaultTemporal.yearMin ||
            temporal.yearMax !== defaultTemporal.yearMax ||
            temporal.monthMin !== defaultTemporal.monthMin ||
            temporal.monthMax !== defaultTemporal.monthMax ||
            temporal.hourMin !== defaultTemporal.hourMin ||
            temporal.hourMax !== defaultTemporal.hourMax
        );
        
        // Check geographic filter
        const hasGeographicFilter = this.state.geographic !== null;
        
        // Check species filter
        const hasSpeciesFilter = this.state.species !== null;
        
        return hasTemporalFilter || hasGeographicFilter || hasSpeciesFilter;
    }
    
    // Reset to default state
    resetToDefaults() {
        this.state = JSON.parse(JSON.stringify(this.defaultState));
        this.saveState();
        this.applyToUI();
        this.notifyListeners('reset', null);
    }
    
    // Get API parameters for filtered queries
    getAPIParams() {
        const params = new URLSearchParams();
        
        // Temporal filters
        params.append('year_min', this.state.temporal.yearMin);
        params.append('year_max', this.state.temporal.yearMax);
        params.append('month_min', this.state.temporal.monthMin);
        params.append('month_max', this.state.temporal.monthMax);
        params.append('hour_min', this.state.temporal.hourMin);
        params.append('hour_max', this.state.temporal.hourMax);
        
        // Geographic filter
        if (this.state.geographic) {
            params.append('north', this.state.geographic.north);
            params.append('south', this.state.geographic.south);
            params.append('east', this.state.geographic.east);
            params.append('west', this.state.geographic.west);
        }
        
        // Species filter
        if (this.state.species) {
            params.append('species', this.state.species);
        }
        
        return params;
    }
}

// Create global instance
window.filterState = new FilterStateManager();

// Export for use
window.FilterStateManager = FilterStateManager;