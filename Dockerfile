FROM python:3.9-slim

# Install git for cloning the repository
RUN apt-get update && apt-get install -y \
    git \
    vim \
    build-essential \
    && apt-get clean

# Set working directory
WORKDIR /app

# Clone the repository
ARG REPO_URL="https://github.com/tansb/StellarPopulation_Synthesis.git"
ARG BRANCH="main"
RUN git clone --depth 1 --branch ${BRANCH} ${REPO_URL} .

# Install the package dependencies from requirements.txt
WORKDIR /app/source/StellarPopulation_Synthesis
RUN pip install --no-cache-dir -r requirements.txt

# Install the package and its dependencies
#RUN pip install --no-cache-dir -e .

# Verify installation
#CMD ["python", "-c", "import StellarPopulation_Synthesis; print('StellarPopulation_Synthesis package successfully installed!')"]