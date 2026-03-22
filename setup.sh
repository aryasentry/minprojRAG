#!/bin/bash

echo "========================================"
echo " Multimodal RAG Course Platform"
echo " Quick Start Script"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "========================================"
echo " Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Ensure Ollama is running: ollama serve"
echo "2. Start backend: python -m src.server"
echo "3. In another terminal, start frontend:"
echo "   cd app"
echo "   npm install"
echo "   npm run dev"
echo "4. Open http://localhost:3000"
echo ""
