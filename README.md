# Autism-Detection
## Problem Statement:
#### AI-Driven Early Detection of Autism in Toddlers Using Multimodal Data
Autism Spectrum Disorder (ASD) is a neurodevelopmental condition that affects communication, behavior, and social interaction. Early diagnosis can significantly improve intervention outcomes. Your task is to design a proof-of-concept AI model or pipeline that aims to detect early behavioral signs of autism in toddlers using non-invasive and multimodal data, particularly focusing on visual cues captured via camera systems.

Your solution should address at least 3 of the following aspects:

Feature Identification: Identify at least three early observable clinical signs of autism (e.g., reduced eye contact, lack of social reciprocity, repetitive movements) that can be detected through video-based technology.

Eye Contact Analysis: Propose a method using camera feed (e.g., webcam or phone camera) to detect reduced or absent eye contact in toddlers during interaction with a caregiver or toy.

Repetitive Behavior Detection: Outline how a machine learning model (e.g., CNN, pose estimation models) could be trained to detect repetitive motor behaviors such as hand flapping or rocking.

Speech-Less Language Delay Detection: In the absence of audio, suggest indirect behavioral proxies (e.g., reduced gesture use, delayed response to stimuli) that AI can analyze to infer possible delayed language development.

Social Reciprocity Assessment: Design a video-based approach to detect limited social interaction, such as the child’s response to name-calling or failure to initiate shared attention (e.g., not pointing or showing toys).
This project is a real-time application that leverages MediaPipe and OpenCV to detect behavioral indicators associated with autism, specifically focusing on repetitive hand motions and eye contact tracking using webcam video feed.

## Solution
This project provides a real-time solution to detect two key behavioral indicators associated with autism—repetitive hand motions and eye contact. By leveraging MediaPipe and OpenCV, it processes webcam feed to identify and visually indicate these behaviors, allowing for potential early intervention and further analysis.

## Features

- Real-time hand flapping detection using pose landmarks.

- Eye contact tracking based on face mesh landmarks.

- Visual indicators for both hand motion and eye contact status.

- Adjustable sensitivity for both motion and eye contact detection.

## Requirements

- Python 3.8+

- OpenCV

- MediaPipe

- NumPy

#### Install the necessary libraries using:

pip install opencv-python-headless mediapipe numpy



## How It Works

### Hand Flapping Detection:

- The application monitors wrist coordinates and calculates standard deviation over a history buffer.

- Repetitive hand motion is detected if the variance in position remains within a defined range over several frames.

### Eye Contact Detection:

- The application calculates the distance of iris landmarks to the eye center to determine focus.

- Eye contact is considered established if both irises remain within a defined threshold for consecutive frames.


## Future Enhancements

- Incorporate additional behavioral indicators (e.g., facial expressions).

- Optimize detection algorithms for faster processing.

- Implement data logging for further analysis.
