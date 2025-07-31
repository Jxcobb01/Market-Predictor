#!/bin/bash

echo "🚀 Setting up Stock Market Predictor..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models data

# Build and start the application
echo "🔨 Building and starting the application..."
docker-compose up --build -d

echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Application is running!"
    echo ""
    echo "🌐 Access the application at:"
    echo "   Frontend: http://localhost:3000"
    echo "   Backend API: http://localhost:8000"
    echo "   API Documentation: http://localhost:8000/docs"
    echo ""
    echo "📋 Next steps:"
    echo "   1. Open http://localhost:3000 in your browser"
    echo "   2. Train the model by visiting http://localhost:8000/api/model/train"
    echo "   3. Start exploring stock predictions!"
    echo ""
    echo "🛑 To stop the application, run: docker-compose down"
else
    echo "❌ Failed to start the application. Check the logs with: docker-compose logs"
    exit 1
fi 