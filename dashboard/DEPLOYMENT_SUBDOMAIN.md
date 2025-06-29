# Deploying DeepEarth Dashboard as a Subdomain

This guide documents the specific steps and considerations when deploying the DeepEarth dashboard as a subdomain on an existing Apache web server.

## Key Learnings from Subdomain Deployment

### 1. Static File Path Conflicts

**Problem**: When deploying on a server that already hosts other sites, the global `/static` alias can conflict with subdomain-specific static files.

**Solution**: Use a unique static path for DeepEarth:
- Changed from `/static/` to `/deepearth-static/` in the Flask app and templates
- This avoids conflicts with the main site's static file handling

### 2. DNS Configuration

**Steps**:
1. Add DNS A record for the subdomain (example using Google Cloud DNS):
   ```bash
   gcloud dns record-sets create subdomain.yourdomain.com. \
     --zone=your-zone-name \
     --type=A \
     --ttl=300 \
     --rrdatas=YOUR_SERVER_IP
   ```

2. Update SSL certificate to include the new subdomain:
   ```bash
   sudo certbot --apache -d yourdomain.com -d subdomain.yourdomain.com --expand
   ```

### 3. Apache Configuration

Add these rules to your Apache SSL configuration:

```apache
# Add subdomain to ServerAlias
ServerAlias subdomain.yourdomain.com

# Static files for subdomain should be served directly
RewriteCond %{HTTP_HOST} ^subdomain\.yourdomain\.com$ [NC]
RewriteCond %{REQUEST_URI} ^/deepearth-static/ [NC]
RewriteRule ^ - [L]

# Subdomain proxy rules
RewriteCond %{HTTP_HOST} ^subdomain\.yourdomain\.com$ [NC]
RewriteRule ^/$ http://127.0.0.1:5003/ [P,L]

RewriteCond %{HTTP_HOST} ^subdomain\.yourdomain\.com$ [NC]
RewriteRule ^/(.+)$ http://127.0.0.1:5003/$1 [P,L]

# Directory permissions for static files
<Directory /path/to/deepearth/dashboard/static>
    Options -Indexes +FollowSymLinks
    AllowOverride None
    Require all granted
</Directory>
```

### 4. Systemd Service Configuration

Create `/etc/systemd/system/deepearth.service`:

```ini
[Unit]
Description=DeepEarth Dashboard Flask Application
After=network.target

[Service]
Type=simple
User=youruser
Group=www-data
WorkingDirectory=/path/to/deepearth/dashboard
Environment=PATH=/path/to/deepearth/dashboard/venv/bin
Environment=FLASK_ENV=production
ExecStart=/path/to/deepearth/dashboard/venv/bin/gunicorn -w 2 -b 127.0.0.1:5003 --timeout 120 --preload deepearth_dashboard:app --access-logfile /path/to/logs/deepearth-access.log --error-logfile /path/to/logs/deepearth-error.log
Restart=always
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

### 5. Common Issues and Solutions

#### Python Command
- **Issue**: Scripts fail with "command not found" for `python`
- **Solution**: Always use `python3` explicitly in all commands and scripts

#### Memory Requirements
- **Issue**: Recommended 32GB RAM, but deployment worked with 15GB
- **Solution**: The mmap conversion will be slower but still functional. Monitor memory usage during conversion.

#### Apache Rewrite Rules
- **Issue**: Missing `$1` in proxy rules causes all paths to redirect to root
- **Solution**: Ensure proxy rules include `$1`: `RewriteRule ^/(.+)$ http://127.0.0.1:5003/$1 [P,L]`

#### SSL Certificate
- **Issue**: Certificate doesn't cover new subdomain
- **Solution**: Expand existing certificate with certbot `--expand` flag

### 6. Deployment Steps Summary

1. **Set up DNS**: Add A record for subdomain
2. **Clone repository**: `git clone https://github.com/legel/deepearth.git`
3. **Create virtual environment**: `python3 -m venv venv`
4. **Install dependencies**: `pip install -r requirements.txt`
5. **Download and convert data**: `python3 prepare_embeddings.py --download deepearth/central-florida-native-plants`
6. **Create systemd service**: Set up service file with port 5003 (or next available)
7. **Update Apache config**: Add subdomain handling and proxy rules
8. **Expand SSL certificate**: Include new subdomain
9. **Start service**: `sudo systemctl enable deepearth.service && sudo systemctl start deepearth.service`
10. **Reload Apache**: `sudo systemctl reload apache2`

### 7. Testing

After deployment, verify:
- Main page loads: `https://subdomain.yourdomain.com/`
- Static files load: `https://subdomain.yourdomain.com/deepearth-static/css/dashboard.css`
- API endpoints work: `https://subdomain.yourdomain.com/api/health`

### 8. Monitoring

Check logs for issues:
- Application logs: `/path/to/logs/deepearth-error.log`
- Access logs: `/path/to/logs/deepearth-access.log`
- Apache logs: `/var/log/apache2/error.log`
- Service status: `sudo systemctl status deepearth.service`

## Alternative: Standalone Deployment

If you're deploying on a dedicated server without existing sites, you can:
1. Keep the standard `/static/` paths (no need to change to `/deepearth-static/`)
2. Use standard port 80/443 without Apache proxy
3. Use simpler nginx configuration instead of Apache

## Security Considerations

1. Ensure the data directory has appropriate permissions
2. Use a dedicated user for the service (not root)
3. Keep the mmap files in a secure location
4. Monitor access logs for unusual activity
5. Consider rate limiting for API endpoints