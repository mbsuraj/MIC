#!/bin/bash

# MIC Web Interface Launcher
echo "🔮 Starting MIC - Make It Certain Web Interface..."

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "Installing web dependencies..."
    pip install -r requirements.txt
fi

# Launch the Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost