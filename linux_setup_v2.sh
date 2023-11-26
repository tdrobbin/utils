#!/bin/bash

# Exit on error
set -e

# Update system packages
echo "Updating system packages..."
sudo apt update

# Install key Ubuntu packages
echo "Installing required packages..."
sudo apt install -y build-essential clang gcc g++ cmake libssl-dev libssh-dev htop libffi-dev wget
sudo apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

# Download and Install Anaconda
ANACONDA_VERSION="2023.09-0"
echo "Installing Anaconda version $ANACONDA_VERSION..."
wget https://repo.anaconda.com/archive/Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh
bash Anaconda3-${ANACONDA_VERSION}-Linux-x86_64.sh -b -p $HOME/anaconda3
echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> $HOME/.bashrc
source $HOME/.bashrc

# Test Anaconda installation
echo "Testing Anaconda installation..."
python -c "print('Anaconda installed successfully')"

# Update Conda and set up libmamba
echo "Setting libmamba as the default solver for conda..."
conda update -n base conda -y
conda install -n base conda-libmamba-solver -y
conda config --set solver libmamba -y

# Install Python 3.10 environment
PY310_ENV_YML="https://raw.githubusercontent.com/tdrobbin/utils/main/py310-environment.yml"
echo "Installing py310 environment from $PY310_ENV_YML"
conda env create --file ${PY310_ENV_YML} -y

# Install Node.js using Conda
echo "Installing Node.js..."
conda install -c conda-forge nodejs -y

# Test Node.js installation
echo "Testing Node.js installation..."
node -e "console.log('Node.js installed successfully')"

# Install Docker
echo "Installing Docker..."
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Test Docker installation
echo "Testing Docker installation..."
sudo docker run hello-world

# Install PostgreSQL
echo "Installing PostgreSQL..."
sudo sh -c 'echo "deb https://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt-get update
sudo apt-get -y install postgresql

# Test PostgreSQL installation
echo "Testing PostgreSQL installation..."
psql --version

echo "Setup completed successfully."
