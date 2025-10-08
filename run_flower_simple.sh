#!/bin/bash

# Simple Flower Runner Script
# Quick start for testing with minimal clients in separate terminal windows

set -e

# Default values
NUM_CLIENTS=${1:-3}
NUM_ROUNDS=${2:-5}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

echo "=========================================="
echo "Flower Simple Runner (Separate Terminals)"
echo "=========================================="
echo "Number of clients: $NUM_CLIENTS"
echo "Number of rounds:  $NUM_ROUNDS"
echo "=========================================="
echo ""

# Cleanup existing processes
print_info "Cleaning up existing Flower processes..."
pkill -f "python.*flower" 2>/dev/null || true
sleep 2

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

# Start server in new terminal
print_info "Starting server in new terminal window..."
open_terminal "Flower Server" "python flower_server.py --num_rounds $NUM_ROUNDS --log_level INFO"

# Wait for server to start
print_info "Waiting for server to start..."
sleep 5

# Start clients in separate terminals
print_info "Starting $NUM_CLIENTS clients in separate terminal windows..."
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    print_info "Starting client $i in new terminal..."
    open_terminal "Flower Client $i" "python flower_client.py --client_id $i --log_level INFO"
    sleep 1
done

print_success "All processes started in separate terminal windows!"
print_warning "Press Ctrl+C in each terminal window to stop individual processes"
print_info "Or run: pkill -f 'python.*flower' to stop all processes"

# Wait for user interrupt
trap 'echo ""; print_warning "Use Ctrl+C in each terminal window to stop processes, or run: pkill -f \"python.*flower\""; exit 0' INT

# Keep script running
while true; do
    sleep 10
done
