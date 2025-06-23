#!/usr/bin/env python3
"""
Deploy DeepSeek-V3 Language Encoder on Google Cloud Platform
"""

import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

class GCloudDeployment:
    """Deploy and manage DeepSeek model on Google Cloud"""
    
    def __init__(self, project_id: str, zone: str = 'us-central1-a'):
        self.project_id = project_id
        self.zone = zone
        self.instance_name = f"deepseek-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
    def create_instance(self, 
                       machine_type: str = 'n2-highmem-80',
                       gpu_type: Optional[str] = None,
                       gpu_count: int = 0,
                       boot_disk_size: int = 500,
                       preemptible: bool = False) -> str:
        """
        Create a GCP instance for DeepSeek model
        
        Args:
            machine_type: GCP machine type (default: n2-highmem-80 with 640GB RAM)
            gpu_type: GPU type if needed (e.g., 'nvidia-tesla-v100')
            gpu_count: Number of GPUs
            boot_disk_size: Boot disk size in GB
            preemptible: Use preemptible instance for cost savings
            
        Returns:
            Instance name
        """
        print(f"Creating instance {self.instance_name}...")
        
        cmd = [
            'gcloud', 'compute', 'instances', 'create', self.instance_name,
            '--project', self.project_id,
            '--zone', self.zone,
            '--machine-type', machine_type,
            '--boot-disk-size', f'{boot_disk_size}GB',
            '--boot-disk-type', 'pd-ssd',
            '--image-family', 'ubuntu-2204-lts',
            '--image-project', 'ubuntu-os-cloud',
            '--metadata-from-file', f'startup-script={self._create_startup_script()}',
            '--scopes', 'cloud-platform',
            '--tags', 'deepseek,http-server,https-server'
        ]
        
        if gpu_type and gpu_count > 0:
            cmd.extend([
                '--accelerator', f'type={gpu_type},count={gpu_count}',
                '--maintenance-policy', 'TERMINATE'
            ])
        
        if preemptible:
            cmd.append('--preemptible')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating instance: {result.stderr}")
            sys.exit(1)
            
        print(f"Instance {self.instance_name} created successfully!")
        
        # Create firewall rule if needed
        self._ensure_firewall_rule()
        
        return self.instance_name
    
    def _create_startup_script(self) -> str:
        """Create startup script for the instance"""
        script_path = Path('/tmp/deepseek_startup.sh')
        
        script_content = '''#!/bin/bash
set -e

# Log all output
exec > >(tee -a /var/log/deepseek-startup.log)
exec 2>&1

echo "Starting DeepSeek setup at $(date)"

# Update system
apt-get update -y
apt-get install -y python3-pip python3-venv git curl wget build-essential

# Create user and directories
if ! id -u photon >/dev/null 2>&1; then
    useradd -m -s /bin/bash photon
fi

mkdir -p /home/photon/deepseek/{models,logs}
chown -R photon:photon /home/photon/deepseek

# Install as photon user
su - photon << 'EOF'
cd /home/photon/deepseek

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install flask requests numpy pandas torch llama-cpp-python

# Download the model if not exists
MODEL_PATH="models/DeepSeek-V3-0324-UD-Q4_K_XL-00001-of-00009.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading DeepSeek model..."
    mkdir -p models
    # Add your model download logic here
    # For example: wget -O "$MODEL_PATH" "YOUR_MODEL_URL"
fi

# Download server script
cat > server.py << 'SCRIPT'
''' + open(Path(__file__).parent / 'server.py').read() + '''
SCRIPT

# Start the server
echo "Starting DeepSeek server..."
nohup python server.py --host 0.0.0.0 --port 8888 > logs/server.log 2>&1 &

echo "Server started. Check logs at /home/photon/deepseek/logs/server.log"
EOF

echo "DeepSeek setup complete at $(date)"
'''
        
        script_path.write_text(script_content)
        return str(script_path)
    
    def _ensure_firewall_rule(self):
        """Ensure firewall rule exists for the model server"""
        rule_name = 'allow-deepseek-server'
        
        # Check if rule exists
        result = subprocess.run(
            ['gcloud', 'compute', 'firewall-rules', 'describe', rule_name,
             '--project', self.project_id],
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"Creating firewall rule {rule_name}...")
            subprocess.run([
                'gcloud', 'compute', 'firewall-rules', 'create', rule_name,
                '--project', self.project_id,
                '--allow', 'tcp:8888',
                '--source-ranges', '0.0.0.0/0',
                '--target-tags', 'deepseek'
            ])
    
    def wait_for_server(self, timeout: int = 600) -> bool:
        """Wait for the server to be ready"""
        print("Waiting for server to be ready...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Get instance IP
            result = subprocess.run([
                'gcloud', 'compute', 'instances', 'describe', self.instance_name,
                '--project', self.project_id,
                '--zone', self.zone,
                '--format', 'get(networkInterfaces[0].accessConfigs[0].natIP)'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                ip = result.stdout.strip()
                if ip:
                    # Try to connect to the server
                    try:
                        import requests
                        response = requests.get(f'http://{ip}:8888/health', timeout=5)
                        if response.status_code == 200:
                            print(f"Server is ready at http://{ip}:8888")
                            return True
                    except:
                        pass
            
            time.sleep(10)
        
        return False
    
    def upload_data(self, local_path: str, remote_path: str):
        """Upload data to the instance"""
        subprocess.run([
            'gcloud', 'compute', 'scp',
            local_path,
            f'{self.instance_name}:{remote_path}',
            '--project', self.project_id,
            '--zone', self.zone
        ])
    
    def run_extraction(self, input_file: str, output_dir: str):
        """Run embedding extraction on the instance"""
        # Get instance IP
        result = subprocess.run([
            'gcloud', 'compute', 'instances', 'describe', self.instance_name,
            '--project', self.project_id,
            '--zone', self.zone,
            '--format', 'get(networkInterfaces[0].accessConfigs[0].natIP)'
        ], capture_output=True, text=True)
        
        ip = result.stdout.strip()
        
        # Create extraction script
        script = f'''
from client import DeepSeekClient, extract_species_embeddings

client = DeepSeekClient('http://localhost:8888')
extract_species_embeddings(client, '{input_file}', '{output_dir}')
'''
        
        # Run on instance
        subprocess.run([
            'gcloud', 'compute', 'ssh', self.instance_name,
            '--project', self.project_id,
            '--zone', self.zone,
            '--command', f'cd /home/photon/deepseek && source venv/bin/activate && python -c "{script}"'
        ])
    
    def download_results(self, remote_path: str, local_path: str):
        """Download results from the instance"""
        subprocess.run([
            'gcloud', 'compute', 'scp',
            '--recurse',
            f'{self.instance_name}:{remote_path}',
            local_path,
            '--project', self.project_id,
            '--zone', self.zone
        ])
    
    def delete_instance(self):
        """Delete the instance"""
        print(f"Deleting instance {self.instance_name}...")
        subprocess.run([
            'gcloud', 'compute', 'instances', 'delete', self.instance_name,
            '--project', self.project_id,
            '--zone', self.zone,
            '--quiet'
        ])

def main():
    parser = argparse.ArgumentParser(description='Deploy DeepSeek on Google Cloud')
    parser.add_argument('--project', required=True, help='GCP project ID')
    parser.add_argument('--zone', default='us-central1-a', help='GCP zone')
    parser.add_argument('--machine-type', default='n2-highmem-80', help='Machine type (default: n2-highmem-80 with 640GB RAM)')
    parser.add_argument('--gpu-type', help='GPU type (e.g., nvidia-tesla-v100)')
    parser.add_argument('--gpu-count', type=int, default=0, help='Number of GPUs')
    parser.add_argument('--disk-size', type=int, default=500, help='Boot disk size in GB')
    parser.add_argument('--preemptible', action='store_true', help='Use preemptible instance')
    parser.add_argument('--data-file', help='Local data file to process')
    parser.add_argument('--output-dir', default='embeddings', help='Output directory')
    parser.add_argument('--keep-instance', action='store_true', help='Keep instance after processing')
    
    args = parser.parse_args()
    
    # Create deployment
    deployment = GCloudDeployment(args.project, args.zone)
    
    try:
        # Create instance
        deployment.create_instance(
            machine_type=args.machine_type,
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
            boot_disk_size=args.disk_size,
            preemptible=args.preemptible
        )
        
        # Wait for server
        if not deployment.wait_for_server():
            print("Server failed to start!")
            sys.exit(1)
        
        # Process data if provided
        if args.data_file:
            # Upload data
            remote_data = f'/home/photon/deepseek/input_data.csv'
            deployment.upload_data(args.data_file, remote_data)
            
            # Run extraction
            deployment.run_extraction(remote_data, f'/home/photon/deepseek/{args.output_dir}')
            
            # Download results
            deployment.download_results(
                f'/home/photon/deepseek/{args.output_dir}',
                args.output_dir
            )
        
        print(f"\nDeployment complete!")
        print(f"Instance: {deployment.instance_name}")
        print(f"Zone: {deployment.zone}")
        
        if not args.keep_instance:
            deployment.delete_instance()
    
    except KeyboardInterrupt:
        print("\nInterrupted. Cleaning up...")
        deployment.delete_instance()
    except Exception as e:
        print(f"Error: {e}")
        deployment.delete_instance()
        sys.exit(1)

if __name__ == '__main__':
    main()