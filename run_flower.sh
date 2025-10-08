#!/bin/bash

# Flower Federated Learning Runner Script
# This script starts one Flower server and multiple Flower clients
# Usage: ./run_flower.sh [num_clients] [num_rounds] [log_level]

set -e  # Exit on any error

# Default values
DEFAULT_NUM_CLIENTS=10
DEFAULT_NUM_ROUNDS=10
DEFAULT_LOG_LEVEL="INFO"
DEFAULT_SERVER_PORT=8080
DEFAULT_CONFIG_NAME="experiments/flower/cifar100_vit_lora/fim/image_cifar100_vit_fedavg_fim-6_9_12-noniid-pat_10_dir-noprior-s50-e50.yaml"

# Parse command line arguments
NUM_CLIENTS=${1:-$DEFAULT_NUM_CLIENTS}
NUM_ROUNDS=${2:-$DEFAULT_NUM_ROUNDS}
LOG_LEVEL=${3:-$DEFAULT_LOG_LEVEL}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to kill existing Flower processes
cleanup_flower_processes() {
    print_info "Cleaning up existing Flower processes..."
    pkill -f "python.*flower" 2>/dev/null || true
    sleep 2
}

# Function to wait for server to be ready
wait_for_server() {
    local max_attempts=30
    local attempt=1
    
    print_info "Waiting for server to be ready..."
    while [ $attempt -le $max_attempts ]; do
        if check_port $DEFAULT_SERVER_PORT; then
            sleep 1
            attempt=$((attempt + 1))
        else
            print_success "Server is ready on port $DEFAULT_SERVER_PORT"
            return 0
        fi
    done
    
    print_error "Server failed to start within 30 seconds"
    return 1
}

# Function to detect terminal emulator
detect_terminal() {
    if command -v osascript >/dev/null 2>&1; then
        echo "osascript"  # macOS Terminal
    elif command -v gnome-terminal >/dev/null 2>&1; then
        echo "gnome-terminal"
    elif command -v xterm >/dev/null 2>&1; then
        echo "xterm"
    elif command -v konsole >/dev/null 2>&1; then
        echo "konsole"
    else
        echo "unknown"
    fi
}

# Function to open terminal window
open_terminal() {
    local title="$1"
    local command="$2"
    local terminal=$(detect_terminal)
    
    case $terminal in
        "osascript")
            osascript -e "tell application \"Terminal\" to do script \"cd '$(pwd)' && echo '=== $title ===' && $command\""
            ;;
        "gnome-terminal")
            gnome-terminal --title="$title" -- bash -c "cd '$(pwd)' && echo '=== $title ===' && $command; exec bash"
            ;;
        "xterm")
            xterm -title "$title" -e bash -c "cd '$(pwd)' && echo '=== $title ===' && $command; exec bash" &
            ;;
        "konsole")
            konsole --new-tab -e bash -c "cd '$(pwd)' && echo '=== $title ===' && $command; exec bash" &
            ;;
        *)
            print_warning "Unknown terminal emulator. Starting processes in background instead."
            eval "$command" &
            ;;
    esac
}

# Function to start the Flower server
start_server() {
    print_info "Starting Flower server in new terminal window..."
    print_info "Configuration: $NUM_ROUNDS rounds, $LOG_LEVEL logging"
    
    # Start server in new terminal window
    open_terminal "Flower Server" "python flower_server.py --num_rounds $NUM_ROUNDS --log_level $LOG_LEVEL --server_port $DEFAULT_SERVER_PORT --config_name $DEFAULT_CONFIG_NAME"
    
    # Wait for server to be ready
    if wait_for_server; then
        print_success "Server started in new terminal window"
        return 0
    else
        print_error "Failed to start server"
        return 1
    fi
}

# Function to start Flower clients
start_clients() {
    print_info "Starting $NUM_CLIENTS Flower clients in separate terminal windows..."
    
    # Start clients in separate terminal windows
    for i in $(seq 0 $((NUM_CLIENTS - 1))); do
        print_info "Starting client $i in new terminal..."
        
        open_terminal "Flower Client $i" "python flower_client.py --client_id $i --log_level $LOG_LEVEL --config_name $DEFAULT_CONFIG_NAME --server_address localhost --server_port $DEFAULT_SERVER_PORT"
        
        # Small delay between client starts
        sleep 1
    done
    
    print_success "All $NUM_CLIENTS clients started in separate terminal windows"
}

# Function to monitor the training
monitor_training() {
    print_info "All processes started in separate terminal windows!"
    print_warning "Press Ctrl+C in each terminal window to stop individual processes"
    print_info "Or run: pkill -f 'python.*flower' to stop all processes"
    
    # Wait for user interrupt
    trap 'echo ""; print_warning "Use Ctrl+C in each terminal window to stop processes, or run: pkill -f \"python.*flower\""; exit 0' INT
    
    # Keep script running
    while true; do
        sleep 10
    done
}

# Function to cleanup and exit
cleanup_and_exit() {
    print_info "Cleaning up processes..."
    
    # Clean up any remaining Flower processes
    cleanup_flower_processes
    
    print_success "Cleanup completed"
    exit 0
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [num_clients] [num_rounds] [log_level]"
    echo ""
    echo "Arguments:"
    echo "  num_clients  Number of clients to start (default: $DEFAULT_NUM_CLIENTS)"
    echo "  num_rounds   Number of training rounds (default: $DEFAULT_NUM_ROUNDS)"
    echo "  log_level    Logging level: DEBUG, INFO, WARNING, ERROR (default: $DEFAULT_LOG_LEVEL)"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start 10 clients, 10 rounds, INFO logging"
    echo "  $0 5                 # Start 5 clients, 10 rounds, INFO logging"
    echo "  $0 5 20              # Start 5 clients, 20 rounds, INFO logging"
    echo "  $0 5 20 DEBUG        # Start 5 clients, 20 rounds, DEBUG logging"
    echo ""
    echo "Terminal Windows:"
    echo "  Each process runs in its own terminal window"
    echo "  Server: 'Flower Server' window"
    echo "  Clients: 'Flower Client X' windows"
}

# Main execution
main() {
    # Check if help is requested
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    # Validate arguments
    if ! [[ "$NUM_CLIENTS" =~ ^[0-9]+$ ]] || [ "$NUM_CLIENTS" -lt 1 ]; then
        print_error "Number of clients must be a positive integer"
        exit 1
    fi
    
    if ! [[ "$NUM_ROUNDS" =~ ^[0-9]+$ ]] || [ "$NUM_ROUNDS" -lt 1 ]; then
        print_error "Number of rounds must be a positive integer"
        exit 1
    fi
    
    if [[ ! "$LOG_LEVEL" =~ ^(DEBUG|INFO|WARNING|ERROR)$ ]]; then
        print_error "Log level must be one of: DEBUG, INFO, WARNING, ERROR"
        exit 1
    fi
    
    # Check if required files exist
    if [ ! -f "flower_server.py" ]; then
        print_error "flower_server.py not found in current directory"
        exit 1
    fi
    
    if [ ! -f "flower_client.py" ]; then
        print_error "flower_client.py not found in current directory"
        exit 1
    fi
    
    if [ ! -f "config/$DEFAULT_CONFIG_NAME" ]; then
        print_error "Configuration file not found: config/$DEFAULT_CONFIG_NAME"
        exit 1
    fi
    
    # Print configuration
    echo "=========================================="
    echo "Flower Federated Learning Runner"
    echo "=========================================="
    echo "Number of clients: $NUM_CLIENTS"
    echo "Number of rounds:  $NUM_ROUNDS"
    echo "Log level:         $LOG_LEVEL"
    echo "Server port:       $DEFAULT_SERVER_PORT"
    echo "Config file:       $DEFAULT_CONFIG_NAME"
    echo "Mode:              Separate Terminal Windows"
    echo "=========================================="
    echo ""
    
    # Cleanup any existing processes
    cleanup_flower_processes
    
    # Start server
    if ! start_server; then
        print_error "Failed to start server"
        exit 1
    fi
    
    # Start clients
    start_clients
    
    # Monitor training
    monitor_training
}

# Run main function
main "$@"
