# Autism-Detection

This project is a real-time application that leverages MediaPipe and OpenCV to detect behavioral indicators associated with autism, specifically focusing on repetitive hand motions and eye contact tracking using webcam video feed.

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
