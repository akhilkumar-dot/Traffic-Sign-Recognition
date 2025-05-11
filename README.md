# Traffic Sign Recognition System

A real-time **Traffic Sign Recognition** system using **Deep Learning** and **FastAPI**. This project utilizes a **Convolutional Neural Network (CNN)** model to classify traffic signs from a webcam stream and displays predictions with confidence in real-time. The frontend is built with **HTML, CSS, and JavaScript**, while the backend is powered by **FastAPI** for real-time WebSocket communication.

## Features

- **Real-time Webcam Stream**: Displays a live stream from the webcam with predictions overlayed.
- **Traffic Sign Classification**: Recognizes traffic signs using a trained CNN model.
- **Confidence Level**: Displays prediction confidence percentage.
- **Voice Alert**: Provides audio feedback when a new prediction is made.
- **Multiple Resolutions**: Supports dynamic webcam resolution based on user configuration.
- **Stop/Start Functionality**: Can start and stop the stream via buttons.
- **FastAPI Backend**: Efficient backend for handling real-time image classification using WebSocket.

## Technologies Used

- **Backend**: FastAPI, WebSocket, OpenCV
- **Frontend**: HTML, CSS, JavaScript
- **Deep Learning**: TensorFlow (Keras)
- **WebSocket**: Real-time communication for streaming video and predictions
- **Audio Alerts**: SpeechSynthesis API (for voice alerts)

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- pip (Python package installer)
- TensorFlow (for model inference)
- FastAPI (for the backend API)
- OpenCV (for handling webcam video capture)
- uvicorn (for FastAPI server)
- JavaScript enabled in your browser (for frontend)

### Install Dependencies

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/traffic-sign-recognition.git
   cd traffic-sign-recognition
