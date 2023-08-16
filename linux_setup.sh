#!/bin/bash

# Exit on error
set -e

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install packages (e.g., git, vim, etc.)
echo "Installing required packages..."
sudo apt install -y build-essential clang gcc g++ cmake libssl-dev libssh-dev htop libffi-dev

# Download and Install Anaconda
echo "Installing Anaconda..."

# Set the Anaconda version
ANACONDA_VERSION="2023.07-2"

# Download Anaconda installer
wget https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh

# Install Anaconda silently (-b)
bash Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -b -p $HOME/anaconda3

# Add Anaconda to PATH
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> $HOME/.bashrc

# Activate the changes to .bashrc
source $HOME/.bashrc

# Set up libmamba as the default solver for conda
echo "Setting libmamba as the default solver for conda..."
conda update -n base conda -y
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba -y

# install py310 environment
PY310_ENV_YML="https://raw.githubusercontent.com/tdrobbin/utils/main/py310-environment.yml"
echo "Installing py310 env:"
echo ${PY310_ENV_YML}
conda env create --file ${PY310_ENV_YML} -y

# Print success message
echo "Setup completed successfully."

# Clean up the Anaconda installer
# rm Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh
