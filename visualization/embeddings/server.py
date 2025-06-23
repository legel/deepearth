#!/usr/bin/env python3
"""
3D Embeddings Visualization Server
Production-ready server for visualizing high-dimensional embeddings in 3D space
"""

import http.server
import socketserver
import os
import json
import argparse
import webbrowser
import logging
import sys
import signal
import mimetypes
from threading import Timer
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from pathlib import Path

# Configure logging
def setup_logging(level=logging.INFO):
    """Configure logging with production-ready format"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class EmbeddingVisualizationHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for embedding visualization"""
    
    def __init__(self, *args, config=None, **kwargs):
        self.config = config or {}
        self.base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        super().__init__(*args, directory=str(self.base_dir), **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            # Route to appropriate handler
            if path == '/config':
                self.serve_config()
            elif path == '/data':
                self.serve_data()
            elif path.startswith('/static/'):
                self.serve_static_file(path)
            elif path == '/health':
                self.serve_health_check()
            elif self.config.get('images', {}).get('enabled') and path == '/image_summary':
                self.serve_image_summary()
            else:
                super().do_GET()
                
        except Exception as e:
            logger.error(f"Error handling GET request: {e}", exc_info=True)
            self.send_error(500, "Internal server error")
    
    def serve_config(self):
        """Serve the current configuration"""
        try:
            # Remove sensitive information before serving
            safe_config = self.sanitize_config(self.config.copy())
            response = json.dumps(safe_config, indent=2)
            
            self.send_json_response(response)
            logger.debug("Served configuration")
            
        except Exception as e:
            logger.error(f"Error serving config: {e}")
            self.send_error(500, "Error serving configuration")
    
    def serve_data(self):
        """Serve the embedding data"""
        try:
            data_file = self.resolve_path(self.config.get('data', {}).get('sourceFile', 'data.json'))
            
            if not data_file.exists():
                logger.error(f"Data file not found: {data_file}")
                self.send_error(404, "Data file not found")
                return
            
            with open(data_file, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(content)
            
            logger.info(f"Served data file: {data_file.name} ({len(content)} bytes)")
            
        except Exception as e:
            logger.error(f"Error serving data: {e}")
            self.send_error(500, "Error serving data")
    
    def serve_static_file(self, path):
        """Serve static files with security checks"""
        try:
            relative_path = path[len('/static/'):]
            static_dir = self.base_dir / 'static'
            requested_path = static_dir / relative_path
            
            # Check if this path is under the biodiversity_images symlink
            if relative_path.startswith('biodiversity_images/'):
                biodiversity_link = static_dir / 'biodiversity_images'
                if biodiversity_link.is_symlink():
                    # This is a valid request through our symlink
                    requested_file = requested_path
                    # Don't resolve() for symlinks - just check if the path exists
                    if not requested_file.exists():
                        logger.warning(f"File not found through symlink: {relative_path}")
                        self.send_error(404, "File not found")
                        return
                    logger.debug(f"Serving biodiversity image: {relative_path}")
                else:
                    logger.error("biodiversity_images symlink not found")
                    self.send_error(404, "Images not configured")
                    return
            else:
                # For regular files, ensure they're within static directory
                requested_file = requested_path.resolve()
                if not str(requested_file).startswith(str(static_dir)):
                    logger.warning(f"Path outside static directory: {path}")
                    self.send_error(403, "Access denied")
                    return
            
            # Handle directory listing
            if requested_file.is_dir():
                if path.endswith('/'):
                    self.list_directory_json(requested_file)
                else:
                    self.send_response(301)
                    self.send_header('Location', path + '/')
                    self.end_headers()
                return
            
            # Serve file
            if not requested_file.exists():
                self.send_error(404, "File not found")
                return
            
            with open(requested_file, 'rb') as f:
                content = f.read()
            
            content_type = mimetypes.guess_type(str(requested_file))[0] or 'application/octet-stream'
            
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.send_header('Cache-Control', 'public, max-age=3600')
            self.end_headers()
            self.wfile.write(content)
            
            logger.debug(f"Served static file: {relative_path}")
            
        except Exception as e:
            logger.error(f"Error serving static file: {e}")
            self.send_error(500, "Error serving file")
    
    def serve_health_check(self):
        """Serve health check endpoint"""
        response = json.dumps({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        })
        self.send_json_response(response)
    
    def serve_image_summary(self):
        """Serve image summary if configured"""
        try:
            summary_file = self.config.get('images', {}).get('summaryFile')
            if not summary_file:
                self.send_error(404, "Image summary not configured")
                return
            
            summary_path = self.resolve_path(summary_file)
            if not summary_path.exists():
                self.send_error(404, "Image summary file not found")
                return
            
            with open(summary_path, 'rb') as f:
                content = f.read()
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
            
            logger.debug("Served image summary")
            
        except Exception as e:
            logger.error(f"Error serving image summary: {e}")
            self.send_error(500, "Error serving image summary")
    
    def list_directory_json(self, directory):
        """List directory contents as JSON"""
        try:
            files = []
            directories = []
            
            for item in sorted(directory.iterdir()):
                if item.is_dir():
                    directories.append(item.name)
                else:
                    files.append(item.name)
            
            response = json.dumps({
                "files": files,
                "directories": directories
            })
            
            self.send_json_response(response)
            
        except Exception as e:
            logger.error(f"Error listing directory: {e}")
            self.send_error(500, "Error listing directory")
    
    def send_json_response(self, content):
        """Send JSON response with proper headers"""
        if isinstance(content, dict):
            content = json.dumps(content)
        
        content_bytes = content.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', len(content_bytes))
        self.end_headers()
        self.wfile.write(content_bytes)
    
    def resolve_path(self, path):
        """Resolve path relative to base directory if not absolute"""
        path = Path(path)
        if not path.is_absolute():
            path = self.base_dir / path
        return path
    
    def sanitize_config(self, config):
        """Remove sensitive information from config"""
        # Remove any paths that might expose system information
        if 'images' in config and 'sourcePath' in config['images']:
            config['images']['sourcePath'] = os.path.basename(config['images']['sourcePath'])
        return config
    
    def end_headers(self):
        """Add security and CORS headers"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('X-XSS-Protection', '1; mode=block')
        super().end_headers()
    
    def log_message(self, format, *args):
        """Override to use logger instead of stderr"""
        logger.debug("%s - %s", self.address_string(), format % args)

class EmbeddingVisualizationServer:
    """Main server class for embedding visualization"""
    
    def __init__(self, config):
        self.config = config
        self.httpd = None
        
    def start(self):
        """Start the server"""
        port = self.config['server']['port']
        host = self.config['server']['host']
        
        # Create handler with config
        handler = lambda *args, **kwargs: EmbeddingVisualizationHandler(
            *args, config=self.config, **kwargs
        )
        
        try:
            self.httpd = socketserver.TCPServer((host, port), handler)
            self.httpd.allow_reuse_address = True
            
            logger.info(f"Server started on {host}:{port}")
            self.print_startup_message()
            
            # Open browser if configured
            if self.config.get('server', {}).get('openBrowser', True):
                Timer(1.0, lambda: webbrowser.open(f'http://{host}:{port}')).start()
            
            # Start serving
            self.httpd.serve_forever()
            
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            if self.httpd:
                self.httpd.shutdown()
                self.httpd.server_close()
    
    def print_startup_message(self):
        """Print startup information"""
        print(f"\n{'='*60}")
        print(f"🌐 3D Embeddings Visualization Server")
        print(f"{'='*60}")
        print(f"Title: {self.config['visualization']['title']}")
        print(f"Data: {self.config['data']['sourceFile']}")
        print(f"Server: http://{self.config['server']['host']}:{self.config['server']['port']}")
        print(f"{'='*60}")
        print("Press Ctrl+C to stop the server\n")

def load_config(config_file):
    """Load configuration from file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise

def create_default_config():
    """Create default configuration"""
    return {
        "visualization": {
            "title": "3D Embeddings Visualization",
            "subtitle": "",
            "defaultSphereSize": 0.05,
            "defaultTextSize": 0.3,
            "scaleFactor": 3.0
        },
        "data": {
            "sourceFile": "data.json",
            "fields": {
                "id": "id",
                "label": "name",
                "cluster": "cluster",
                "metadata": []
            }
        },
        "images": {
            "enabled": False
        },
        "server": {
            "host": "localhost",
            "port": 8080,
            "openBrowser": True
        }
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='3D Embeddings Visualization Server',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default configuration
  python server.py --data embeddings.json
  
  # Use custom configuration file
  python server.py --config config.json
  
  # Override specific settings
  python server.py --config config.json --port 8090 --title "My Embeddings"
        """
    )
    
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data', type=str, help='Data file path (overrides config)')
    parser.add_argument('--port', type=int, help='Server port (overrides config)')
    parser.add_argument('--host', type=str, help='Server host (overrides config)')
    parser.add_argument('--title', type=str, help='Visualization title (overrides config)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    try:
        # Load configuration
        if args.config:
            config = load_config(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config = create_default_config()
            logger.info("Using default configuration")
        
        # Override with command line arguments
        if args.data:
            config['data']['sourceFile'] = args.data
        if args.port:
            config['server']['port'] = args.port
        if args.host:
            config['server']['host'] = args.host
        if args.title:
            config['visualization']['title'] = args.title
        if args.no_browser:
            config['server']['openBrowser'] = False
        
        # Validate data file exists
        data_file = Path(config['data']['sourceFile'])
        if not data_file.is_absolute():
            data_file = Path(__file__).parent / data_file
        
        if not data_file.exists():
            logger.error(f"Data file not found: {data_file}")
            return 1
        
        # Create and start server
        server = EmbeddingVisualizationServer(config)
        server.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
        return 0
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())