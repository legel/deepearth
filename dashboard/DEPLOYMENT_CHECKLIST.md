# DeepEarth Dashboard Deployment Checklist

This checklist ensures a smooth deployment of the DeepEarth Multimodal Geospatial Dashboard.

## Pre-Deployment Setup

### 1. System Requirements
- [ ] Python 3.8 or higher installed
- [ ] 32GB+ RAM available
- [ ] 250GB+ disk space (for embeddings.mmap file)
- [ ] Modern web browser with WebGL support

### 2. Download Dataset
- [ ] Download from HuggingFace and prepare embeddings:
  ```bash
  python prepare_embeddings.py --download deepearth/central-florida-native-plants
  ```
- [ ] Verify dataset structure includes:
  - [ ] `observations.parquet`
  - [ ] `vision_embeddings/` directory with 159 parquet files
  - [ ] `vision_index.parquet`
  - [ ] `dataset_info.json`

### 3. Prepare Memory-Mapped Embeddings
- [ ] The embedding conversion happens automatically after download (~50 minutes)
- [ ] Verify output files created:
  - [ ] `embeddings.mmap` (~206GB)
  - [ ] `embeddings_index.db` (SQLite index)

### 4. Configuration
- [ ] Configuration is automatically set to use `./huggingface_dataset`
- [ ] Cache directory will be created automatically

### 5. Install Dependencies
- [ ] Create virtual environment:
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- [ ] Install requirements:
  ```bash
  pip install -r requirements.txt
  ```

## Validation

### 6. Run Setup Validation
- [ ] Execute validation script:
  ```bash
  python setup_dashboard.py
  ```
- [ ] Address any reported issues
- [ ] All critical checks should pass

### 7. Test Dashboard
- [ ] Start in development mode:
  ```bash
  python deepearth_dashboard.py
  ```
- [ ] Open http://localhost:5000 in browser
- [ ] Verify:
  - [ ] Map loads with observation points
  - [ ] 3D visualizations work
  - [ ] Can click on observations for details
  - [ ] Vision attention maps display correctly

## Production Deployment

### 8. Configure for Production
- [ ] Set up reverse proxy (Nginx/Apache)
- [ ] Configure SSL certificates
- [ ] Set appropriate file permissions
- [ ] Configure firewall rules

### 9. Deploy with Gunicorn
- [ ] Use production startup script:
  ```bash
  ./run_production.sh
  ```
- [ ] Or manually:
  ```bash
  gunicorn -w 4 -b 0.0.0.0:5000 --timeout 120 deepearth_dashboard:app
  ```

### 10. Set up Process Management
- [ ] Create systemd service (Linux) or equivalent
- [ ] Enable auto-start on boot
- [ ] Configure logging

## Post-Deployment

### 11. Performance Optimization
- [ ] Warm up cache by accessing common queries
- [ ] Monitor memory usage
- [ ] Check `/api/health` endpoint

### 12. Monitoring
- [ ] Set up application monitoring
- [ ] Configure error logging
- [ ] Set up alerts for issues

## Troubleshooting

### Common Issues

**"Too many open files" error**
```bash
ulimit -n 65536
```

**Memory-mapped file not loading**
- Check file permissions
- Verify file path in logs
- Ensure sufficient memory

**Slow performance**
- First access is slower (OS page cache)
- Check cache hit rates at `/api/health`
- Consider SSD storage for mmap file

**Missing dependencies**
```bash
pip install -r requirements.txt --upgrade
```

## Migration to deepearth/dashboard

When moving to the final repository location:

1. [ ] Copy entire deployment directory
2. [ ] Update any absolute paths in configuration
3. [ ] Re-run `setup_dashboard.py` to validate
4. [ ] Update any external references/links

---

✅ Once all items are checked, your DeepEarth Dashboard is ready for use!