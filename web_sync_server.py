#!/usr/bin/env python3
"""
Web Sync Server for SAR Drone AI System
Provides web interface for real-time code synchronization
"""

import os
import json
import time
import hashlib
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import logging

# Web framework
try:
    from flask import Flask, render_template, jsonify, send_file, request
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not available, using simple HTTP server")

# Simple HTTP server fallback
import http.server
import socketserver
import urllib.parse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FileTracker:
    """Track file changes and generate hashes"""
    
    def __init__(self, project_path: str = "."):
        self.project_path = Path(project_path).resolve()
        self.file_hashes = {}
        self.last_scan = None
        self.ignored_patterns = {
            '__pycache__', '.git', '.vscode', '.idea',
            '*.log', '*.db', '*.sqlite', '*.tmp',
            'logs/', 'data/', 'portable_ai_system/',
            'node_modules/', 'venv/', 'env/'
        }
        
    def _should_ignore(self, file_path: Path) -> bool:
        """Check if file should be ignored"""
        rel_path = file_path.relative_to(self.project_path)
        
        # Check exact matches
        if rel_path.name in self.ignored_patterns:
            return True
            
        # Check patterns
        for pattern in self.ignored_patterns:
            if pattern.endswith('/') and str(rel_path).startswith(pattern[:-1]):
                return True
            if pattern.startswith('*.') and rel_path.suffix == pattern[1:]:
                return True
                
        return False
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Get MD5 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""
    
    def scan_files(self) -> Dict[str, Any]:
        """Scan all files and return changes"""
        changes = {
            'new_files': [],
            'modified_files': [],
            'deleted_files': [],
            'unchanged_files': [],
            'total_files': 0
        }
        
        current_files = set()
        
        # Scan all files
        for file_path in self.project_path.rglob('*'):
            if file_path.is_file() and not self._should_ignore(file_path):
                rel_path = str(file_path.relative_to(self.project_path))
                current_files.add(rel_path)
                
                current_hash = self._get_file_hash(file_path)
                
                if rel_path not in self.file_hashes:
                    # New file
                    changes['new_files'].append({
                        'path': rel_path,
                        'hash': current_hash,
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                elif self.file_hashes[rel_path] != current_hash:
                    # Modified file
                    changes['modified_files'].append({
                        'path': rel_path,
                        'hash': current_hash,
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
                else:
                    # Unchanged file
                    changes['unchanged_files'].append(rel_path)
                
                self.file_hashes[rel_path] = current_hash
        
        # Find deleted files
        for old_file in set(self.file_hashes.keys()) - current_files:
            changes['deleted_files'].append(old_file)
            del self.file_hashes[old_file]
        
        changes['total_files'] = len(current_files)
        self.last_scan = datetime.now()
        
        return changes

class WebSyncServer:
    """Web server for file synchronization"""
    
    def __init__(self, project_path: str = ".", port: int = 8080, sync_interval: int = 30):
        self.project_path = Path(project_path).resolve()
        self.port = port
        self.sync_interval = sync_interval
        self.file_tracker = FileTracker(project_path)
        self.changes_history = []
        self.last_sync = None
        self.sync_thread = None
        self.running = False
        
        # Initialize Flask app if available
        if FLASK_AVAILABLE:
            self.app = Flask(__name__)
            CORS(self.app)
            self._setup_flask_routes()
        else:
            self.app = None
    
    def _setup_flask_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/status')
        def status():
            return jsonify({
                'project_path': str(self.project_path),
                'last_sync': self.last_sync.isoformat() if self.last_sync else None,
                'total_files': len(self.file_tracker.file_hashes),
                'recent_changes': self.changes_history[-10:] if self.changes_history else []
            })
        
        @self.app.route('/api/changes')
        def changes():
            changes = self.file_tracker.scan_files()
            if any([changes['new_files'], changes['modified_files'], changes['deleted_files']]):
                self.changes_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'changes': changes
                })
                self.last_sync = datetime.now()
            return jsonify(changes)
        
        @self.app.route('/api/files')
        def files():
            files_info = []
            for rel_path, file_hash in self.file_tracker.file_hashes.items():
                file_path = self.project_path / rel_path
                if file_path.exists():
                    files_info.append({
                        'path': rel_path,
                        'hash': file_hash,
                        'size': file_path.stat().st_size,
                        'modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                    })
            return jsonify(files_info)
        
        @self.app.route('/api/download/<path:file_path>')
        def download_file(file_path):
            full_path = self.project_path / file_path
            if full_path.exists() and full_path.is_file():
                return send_file(full_path, as_attachment=True)
            return jsonify({'error': 'File not found'}), 404
        
        @self.app.route('/api/download-all')
        def download_all():
            # Create a zip file of all project files
            import zipfile
            import tempfile
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for rel_path in self.file_tracker.file_hashes.keys():
                    file_path = self.project_path / rel_path
                    if file_path.exists():
                        zipf.write(file_path, rel_path)
            
            return send_file(temp_file.name, as_attachment=True, download_name='sar_drone_ai_project.zip')
    
    def _sync_worker(self):
        """Background worker for file synchronization"""
        while self.running:
            try:
                changes = self.file_tracker.scan_files()
                if any([changes['new_files'], changes['modified_files'], changes['deleted_files']]):
                    self.changes_history.append({
                        'timestamp': datetime.now().isoformat(),
                        'changes': changes
                    })
                    self.last_sync = datetime.now()
                    logger.info(f"Sync completed: {len(changes['new_files'])} new, {len(changes['modified_files'])} modified, {len(changes['deleted_files'])} deleted")
                
                time.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Sync error: {e}")
                time.sleep(self.sync_interval)
    
    def start(self):
        """Start the web server"""
        self.running = True
        
        # Start sync thread
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        logger.info(f"Starting web sync server on port {self.port}")
        logger.info(f"Project path: {self.project_path}")
        logger.info(f"Sync interval: {self.sync_interval} seconds")
        
        if FLASK_AVAILABLE:
            # Create templates directory and HTML template
            self._create_templates()
            self.app.run(host='0.0.0.0', port=self.port, debug=False)
        else:
            # Use simple HTTP server
            self._start_simple_server()
    
    def _create_templates(self):
        """Create HTML template for web interface"""
        templates_dir = Path('templates')
        templates_dir.mkdir(exist_ok=True)
        
        html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAR Drone AI - Web Sync</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .changes { margin-bottom: 20px; }
        .file-list { max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; }
        .file-item { padding: 5px; border-bottom: 1px solid #eee; }
        .file-item:hover { background: #f9f9f9; }
        .new { color: green; }
        .modified { color: orange; }
        .deleted { color: red; }
        .btn { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
        .btn:hover { background: #0056b3; }
        .refresh { background: #28a745; }
        .download { background: #17a2b8; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚁 SAR Drone AI System</h1>
            <p>Real-time Code Synchronization</p>
        </div>
        
        <div class="status" id="status">
            <h3>System Status</h3>
            <p>Loading...</p>
        </div>
        
        <div class="changes" id="changes">
            <h3>Recent Changes</h3>
            <button class="btn refresh" onclick="refreshChanges()">🔄 Refresh</button>
            <button class="btn download" onclick="downloadAll()">📦 Download All</button>
            <div id="changes-content">Loading...</div>
        </div>
        
        <div class="file-list" id="file-list">
            <h3>Project Files</h3>
            <div id="files-content">Loading...</div>
        </div>
    </div>

    <script>
        function refreshStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').innerHTML = `
                        <h3>System Status</h3>
                        <p><strong>Project Path:</strong> ${data.project_path}</p>
                        <p><strong>Total Files:</strong> ${data.total_files}</p>
                        <p><strong>Last Sync:</strong> ${data.last_sync || 'Never'}</p>
                    `;
                });
        }
        
        function refreshChanges() {
            fetch('/api/changes')
                .then(response => response.json())
                .then(data => {
                    let content = '';
                    if (data.new_files.length > 0) {
                        content += '<h4 class="new">📄 New Files:</h4><ul>';
                        data.new_files.forEach(file => {
                            content += `<li>${file.path} (${file.size} bytes)</li>`;
                        });
                        content += '</ul>';
                    }
                    if (data.modified_files.length > 0) {
                        content += '<h4 class="modified">✏️ Modified Files:</h4><ul>';
                        data.modified_files.forEach(file => {
                            content += `<li>${file.path} (${file.size} bytes)</li>`;
                        });
                        content += '</ul>';
                    }
                    if (data.deleted_files.length > 0) {
                        content += '<h4 class="deleted">🗑️ Deleted Files:</h4><ul>';
                        data.deleted_files.forEach(file => {
                            content += `<li>${file}</li>`;
                        });
                        content += '</ul>';
                    }
                    if (content === '') {
                        content = '<p>No changes detected</p>';
                    }
                    document.getElementById('changes-content').innerHTML = content;
                });
        }
        
        function refreshFiles() {
            fetch('/api/files')
                .then(response => response.json())
                .then(data => {
                    let content = '';
                    data.forEach(file => {
                        content += `<div class="file-item">
                            <strong>${file.path}</strong> 
                            <span style="color: #666;">(${file.size} bytes, modified: ${file.modified})</span>
                            <button class="btn download" onclick="downloadFile('${file.path}')" style="float: right; padding: 2px 8px; font-size: 12px;">Download</button>
                        </div>`;
                    });
                    document.getElementById('files-content').innerHTML = content;
                });
        }
        
        function downloadFile(filePath) {
            window.open(`/api/download/${filePath}`, '_blank');
        }
        
        function downloadAll() {
            window.open('/api/download-all', '_blank');
        }
        
        // Auto-refresh every 30 seconds
        setInterval(() => {
            refreshStatus();
            refreshChanges();
            refreshFiles();
        }, 30000);
        
        // Initial load
        refreshStatus();
        refreshChanges();
        refreshFiles();
    </script>
</body>
</html>'''
        
        with open(templates_dir / 'index.html', 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _start_simple_server(self):
        """Start simple HTTP server as fallback"""
        class SyncHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(self.project_path), **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    html = f'''<!DOCTYPE html>
<html>
<head><title>SAR Drone AI - File Browser</title></head>
<body>
<h1>🚁 SAR Drone AI System</h1>
<p>Project files are available for download. Navigate to view files.</p>
<p><strong>Project Path:</strong> {self.project_path}</p>
<p><strong>Last Sync:</strong> {self.last_sync.isoformat() if self.last_sync else 'Never'}</p>
</body>
</html>'''
                    self.wfile.write(html.encode())
                else:
                    super().do_GET()
        
        with socketserver.TCPServer(("", self.port), SyncHTTPRequestHandler) as httpd:
            logger.info(f"Simple HTTP server running on port {self.port}")
            httpd.serve_forever()
    
    def stop(self):
        """Stop the web server"""
        self.running = False
        logger.info("Web sync server stopped")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Sync Server for SAR Drone AI")
    parser.add_argument("--port", type=int, default=8080, help="Port to run server on")
    parser.add_argument("--path", default=".", help="Project path to sync")
    parser.add_argument("--interval", type=int, default=30, help="Sync interval in seconds")
    
    args = parser.parse_args()
    
    try:
        server = WebSyncServer(args.path, args.port, args.interval)
        print(f"🚁 Starting SAR Drone AI Web Sync Server")
        print(f"📁 Project: {args.path}")
        print(f"🌐 Web Interface: http://localhost:{args.port}")
        print(f"⏱️  Sync Interval: {args.interval} seconds")
        print(f"📋 Press Ctrl+C to stop")
        
        server.start()
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 