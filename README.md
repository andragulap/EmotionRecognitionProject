# Multimodal Emotion Recognition Project

**Status: Work in Progress**

This repository contains the implementation of a multimodal system for human emotion recognition. The project integrates audio and video data to improve classification accuracy compared to single-modality approaches.

## System Architecture

The project is built on a late-fusion strategy involving two specialized neural networks:
- **Audio Model**:  trained on vocal features extracted from speech signals;
- **Video Model**: focused on facial expression analysis and micro-expressions;
- **Fusion Layer**:  combines representations from both modalities to produce an emotion prediction;

### Datasets Used
- **RAVDESS**: The Ryerson Audio-Visual Database of Emotional Speech and Song;
- **CREMA-D**: Crowd-sourced Emotional Multimodal Actors Dataset;

## Project Structure
- `audio/`: Python scripts for audio preprocessing and audio model training;
- `video/`: (Coming soon) Scripts for frame extraction, face detection, and video model training;
- `fusion/`: (Coming soon) Integration logic and multimodal fusion layers;
- `requirements.txt`: List of dependencies and libraries required to run the project;
- `.gitignore`: Configuration file to exclude large datasets, video frames, and compiled model files;

## Experimental Results

### Audio Modality Performance
Below is the confusion matrix for the current audio-only emotion recognition model:

<img width="1128" height="898" alt="image" src="https://github.com/user-attachments/assets/9032db74-bb12-4378-bc8b-cd89622be8f8" />

## Installation and Usage
1. Clone the repository to your local machine.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```




