FROM python:3.9-slim

# Install git for cloning the repository
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    vim \
    build-essential \
    && apt-get clean

# Clone the repository
RUN git clone "https://github.com/tansb/StellarProperties.git"
# get the large files via lfs
WORKDIR /StellarProperties
RUN git lfs install
RUN git lfs pull

# Install the package dependencies from requirements.txt
RUN pip install -r requirements.txt

# Add the source dir to the python path so
RUN export PYTHONPATH="${PYTHONPATH}:/StellarProperties/source/StellarPopulation_Synthesis"


# Verify installation
CMD ["python", "-c", "import Galaxy; print('Galaxy package successfully installed!')"]