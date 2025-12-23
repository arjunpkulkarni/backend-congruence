#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Determine docker compose command
if docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Function to build and start containers
deploy_docker() {
    print_info "Building Docker image..."
    $DOCKER_COMPOSE build
    
    print_info "Starting containers..."
    $DOCKER_COMPOSE up -d
    
    print_info "Deployment complete!"
    print_info "API is running at http://localhost:8000"
    print_info "Health check: http://localhost:8000/health"
    print_info ""
    print_info "To view logs, run: $DOCKER_COMPOSE logs -f"
    print_info "To stop, run: $DOCKER_COMPOSE down"
}

# Function to stop containers
stop_docker() {
    print_info "Stopping containers..."
    $DOCKER_COMPOSE down
    print_info "Containers stopped."
}

# Function to restart containers
restart_docker() {
    print_info "Restarting containers..."
    $DOCKER_COMPOSE restart
    print_info "Containers restarted."
}

# Function to show logs
logs_docker() {
    $DOCKER_COMPOSE logs -f
}

# Function to rebuild and restart
rebuild_docker() {
    print_info "Rebuilding containers..."
    $DOCKER_COMPOSE down
    $DOCKER_COMPOSE build --no-cache
    $DOCKER_COMPOSE up -d
    print_info "Rebuild complete!"
}

# Main script
case "${1:-docker}" in
    docker|start)
        deploy_docker
        ;;
    stop)
        stop_docker
        ;;
    restart)
        restart_docker
        ;;
    logs)
        logs_docker
        ;;
    rebuild)
        rebuild_docker
        ;;
    *)
        print_info "Usage: $0 {docker|start|stop|restart|logs|rebuild}"
        print_info "  docker/start  - Build and start containers"
        print_info "  stop          - Stop containers"
        print_info "  restart       - Restart containers"
        print_info "  logs          - Show container logs"
        print_info "  rebuild       - Rebuild and restart containers"
        exit 1
        ;;
esac




