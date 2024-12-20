#!/bin/bash

# Create and activate virtual environment
python -m venv albumify_env
source albumify_env/bin/activate

# Install pip requirements
pip install -r requirements.txt

# Download dlib face landmark predictor
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# Create necessary directories
mkdir -p model_checkpoints
mkdir -p logs