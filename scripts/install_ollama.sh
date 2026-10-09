#!/bin/bash

# Cleanup any previous installation
echo "Removing previous installation of ollama..."
rm -rf $HOME/ollama
rm -rf $HOME/.local/bin/ollama
rm -rf $HOME/.local/lib/ollama
rm -rf $HOME/.local/ollama

# Create folder and export paths
echo "Creating necessary directories and setting environment variables..."
mkdir -p $HOME/.local/bin
mkdir -p $HOME/.local/lib

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="$HOME/.local/lib/ollama:$LD_LIBRARY_PATH"' >> ~/.bashrc

# Download and extract ollama
echo "Downloading and installing ollama..."
cd $HOME
mkdir ollama
cd ollama
curl -L "https://github.com/ollama/ollama/releases/download/v0.13.5/ollama-linux-amd64.tgz" -o ollama.tgz 
tar -xzf ollama.tgz

# Move binaries and libraries to appropriate locations
echo "Finalizing installation..."
cd $HOME
mv ollama/bin/ollama $HOME/.local/bin/ollama
chmod +x $HOME/.local/bin/ollama

mv ollama/lib/ollama $HOME/.local/lib/

# Cleanup
echo "Cleaning up temporary files..."
rm -rf $HOME/ollama
