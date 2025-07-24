#!/bin/bash

# DeepEarth MLP U-Net Training Pipeline
# This script runs the complete pipeline with nohup for background execution

# Set script name and paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs

# Generate timestamp for unique log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs/run_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Function to print colored output
print_status() {
    echo -e "\033[1;32m[$(date +'%Y-%m-%d %H:%M:%S')]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[$(date +'%Y-%m-%d %H:%M:%S')]\033[0m $1"
}

# Check if virtual environment exists
if [ ! -d "../../venv_mlp_unet" ]; then
    print_error "Virtual environment not found! Please run setup first."
    exit 1
fi

# Function to run a command with nohup and monitor
run_with_nohup() {
    local cmd_name=$1
    local cmd=$2
    local log_file="$LOG_DIR/${cmd_name}.log"
    local pid_file="$LOG_DIR/${cmd_name}.pid"
    
    print_status "Starting $cmd_name..."
    
    # Run command with nohup
    nohup bash -c "source ../../venv_mlp_unet/bin/activate && $cmd" > "$log_file" 2>&1 &
    local pid=$!
    echo $pid > "$pid_file"
    
    print_status "$cmd_name started with PID: $pid"
    print_status "Log file: $log_file"
    
    # Wait for the process to complete
    while kill -0 $pid 2>/dev/null; do
        sleep 10
        # Show last line of log
        if [ -f "$log_file" ]; then
            last_line=$(tail -n 1 "$log_file" | tr -d '\n' | cut -c1-80)
            echo -ne "\r\033[K[$cmd_name] $last_line..."
        fi
    done
    
    echo ""  # New line after progress
    
    # Check exit status
    wait $pid
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        print_status "$cmd_name completed successfully!"
        return 0
    else
        print_error "$cmd_name failed with exit code: $exit_code"
        print_error "Check log file: $log_file"
        return 1
    fi
}

# Main pipeline
main() {
    print_status "=== DeepEarth MLP U-Net Training Pipeline ==="
    print_status "All logs will be saved to: $LOG_DIR"
    
    # Step 1: Download images if needed
    if [ ! -d "data/plants" ] || [ ! -f "data/plants/species_list.txt" ]; then
        print_status "Step 1: Downloading images..."
        if ! run_with_nohup "download" "python download_images.py"; then
            print_error "Download failed! Exiting."
            exit 1
        fi
    else
        print_status "Step 1: Images already downloaded, skipping..."
    fi
    
    # Step 2: Quick test with quickstart
    print_status "Step 2: Running quickstart test..."
    if ! run_with_nohup "quickstart" "python quickstart.py --epochs 5"; then
        print_error "Quickstart failed! Exiting."
        exit 1
    fi
    
    # Step 3: Full training
    print_status "Step 3: Running full training..."
    if ! run_with_nohup "training" "python train.py"; then
        print_error "Training failed! Exiting."
        exit 1
    fi
    
    # Step 4: Run inference/visualization
    print_status "Step 4: Running inference and visualization..."
    if ! run_with_nohup "inference" "python inference.py"; then
        print_error "Inference failed! Continuing anyway..."
    fi
    
    # Summary
    print_status "=== Pipeline Complete ==="
    print_status "Logs saved to: $LOG_DIR"
    print_status "Checkpoints saved to: checkpoints/"
    print_status "Visualizations saved to: visualizations/"
    
    # Show final model performance if available
    if [ -f "$LOG_DIR/training.log" ]; then
        print_status "Final training results:"
        grep -E "(accuracy|loss)" "$LOG_DIR/training.log" | tail -5
    fi
}

# Create a wrapper script that can be run with nohup
create_wrapper() {
    cat > "$LOG_DIR/pipeline_wrapper.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
cd ../..
./run_pipeline.sh
EOF
    chmod +x "$LOG_DIR/pipeline_wrapper.sh"
}

# If script is called with 'nohup' argument, run the entire pipeline in background
if [ "$1" == "nohup" ]; then
    create_wrapper
    print_status "Starting entire pipeline in background..."
    nohup "$LOG_DIR/pipeline_wrapper.sh" > "$LOG_DIR/pipeline_main.log" 2>&1 &
    pid=$!
    echo $pid > "$LOG_DIR/pipeline_main.pid"
    print_status "Pipeline started with PID: $pid"
    print_status "Main log: $LOG_DIR/pipeline_main.log"
    print_status "Monitor with: tail -f $LOG_DIR/pipeline_main.log"
    exit 0
fi

# Run main pipeline
main
