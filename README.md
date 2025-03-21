# 🏋️‍♂️ Pose Classification Project (ADE)  

This project is a **real-time pose classification system** using **Python, OpenCV, and MediaPipe**. It detects and classifies human poses, making it useful for fitness tracking, posture correction, and AI-powered surveillance.

## 🚀 Technologies Used  

- **Python** – Core programming language  
- **Flask** – Web framework for serving the model  
- **OpenCV** – Image processing and computer vision  
- **MediaPipe** – Human pose detection  
- **TensorFlow** – Optional GPU acceleration (Mac MPS support)  
- **Docker** – Containerization for easy deployment  

## 📂 Project Structure  

📁 Pose_Classification_ADE
┣ 📂 static/images # Sample images for testing
┃ ┣ 📜 s.jpeg
┃ ┣ 📜 sit1.jpeg
┃ ┣ 📜 side1.jpeg
┃ ┗ 📜 ...
┣ 📂 templates # HTML templates for web interface
┃ ┗ 📜 index.html
┣ 📜 app.py # Flask app for serving the model
┣ 📜 pose_detection.py # Main pose classification logic
┣ 📜 requirements.txt # Dependencies
┣ 📜 Dockerfile # Docker container configuration
┣ 📜 README.md # Project documentation
┗ 📂 pycache # Compiled Python files


## 📌 Features  

✅ **Real-time Pose Detection** – Detects and visualizes key points  
✅ **Pose Classification** – Recognizes standing, sitting, and other postures  
✅ **Angle Calculation** – Computes joint angles for accurate classification  
✅ **Web Interface (Flask)** – Serve predictions via a simple UI  
✅ **Docker Support** – Run the project in an isolated container  

## ▶️ Getting Started  

### 1️ Install Dependencies  

```bash
pip install -r requirements.txt

### 2.Run Application  
python app.py

### 2.Run with Docker
docker build -t pose-classification .
docker run -p 5000:5000 pose-classification
  


