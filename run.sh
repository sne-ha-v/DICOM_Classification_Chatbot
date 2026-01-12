#!/bin/bash

# Lung Nodule AI Assistant - One-Click Runner
# This script starts both the FastAPI backend and Streamlit frontend

echo "🫁 Lung Nodule AI Assistant - Starting Services..."
echo "=================================================="

# Function to check if port is in use
check_port() {
    lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1
    return $?
}

# Function to kill process on port
kill_port() {
    local port=$1
    local pids=$(lsof -ti:$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        echo "🔄 Killing existing processes on port $port (PIDs: $pids)..."
        for pid in $pids; do
            kill $pid 2>/dev/null
        done
        sleep 3
        # Force kill if still running
        if check_port $port; then
            for pid in $pids; do
                kill -9 $pid 2>/dev/null
            done
            sleep 2
        fi
    fi
}

# Function to wait for port to be free
wait_for_port_free() {
    local port=$1
    local max_attempts=10
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if ! check_port $port; then
            return 0
        fi
        echo "   Waiting for port $port to be free... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    return 1
}

# Check if virtual environment exists and activate it
if [ -d ".venv" ]; then
    echo "🐍 Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if required files exist
if [ ! -f "run_api.py" ]; then
    echo "❌ run_api.py not found!"
    exit 1
fi

if [ ! -f "app.py" ]; then
    echo "❌ app.py not found!"
    exit 1
fi

# Clean up ports before starting
echo "🧹 Checking for port conflicts..."
kill_port 8000
kill_port 8501

# Wait for ports to be free
echo "⏳ Ensuring ports are available..."
if ! wait_for_port_free 8000; then
    echo "❌ Could not free port 8000"
    exit 1
fi
if ! wait_for_port_free 8501; then
    echo "❌ Could not free port 8501"
    exit 1
fi
echo "✅ Ports cleared and available"

# Start FastAPI backend in background
echo "🚀 Starting FastAPI backend on port 8000..."
source .venv/bin/activate && python run_api.py &
API_PID=$!

# Wait for API to start with better checking
echo "⏳ Waiting for FastAPI to start..."
max_attempts=15
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ FastAPI backend is running (attempt $attempt)"
        break
    fi
    echo "   Waiting for FastAPI... (attempt $attempt/$max_attempts)"
    sleep 2
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ FastAPI failed to start after $max_attempts attempts"
    echo "   Check the logs above for errors"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Start Streamlit frontend
echo "🌐 Starting Streamlit frontend on port 8501..."
source .venv/bin/activate && streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

# Wait for Streamlit to start
echo "⏳ Waiting for Streamlit to start..."
max_attempts=10
attempt=1
while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo "✅ Streamlit frontend is running (attempt $attempt)"
        break
    fi
    echo "   Waiting for Streamlit... (attempt $attempt/$max_attempts)"
    sleep 2
    ((attempt++))
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ Streamlit failed to start after $max_attempts attempts"
    echo "   Check the logs above for errors"
    kill $STREAMLIT_PID 2>/dev/null
    kill $API_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Services started successfully!"
echo "=================================================="
echo "📊 FastAPI Backend: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🖥️  Streamlit Frontend: http://localhost:8501"
echo ""
echo "💡 To stop all services: Ctrl+C or run 'pkill -f \"python run_api.py\" && pkill -f streamlit'"
echo "=================================================="

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    kill $STREAMLIT_PID 2>/dev/null
    kill $API_PID 2>/dev/null
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
echo "Press Ctrl+C to stop all services..."
wait