import math
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# Set up MPS for accelerated inference if available
if tf.config.list_physical_devices('GPU'):
    tf.config.set_visible_devices([], 'GPU')
    mps_device = tf.config.experimental.list_physical_devices('MPS')
    if mps_device:
        tf.config.experimental.set_visible_devices(mps_device, 'MPS')
        print("MPS device set for acceleration")

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Initialize the pose function and video capture.
def init_pose():
    return mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def detect_pose(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    bbox = None

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        x_coords, y_coords = [], []

        for landmark in results.pose_landmarks.landmark:
            x_coords.append(int(landmark.x * width))
            y_coords.append(int(landmark.y * height))
            landmarks.append((int(landmark.x * width), int(landmark.y * height), landmark.z * width))

        bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

    return output_image, landmarks, bbox

def calculate_angle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    return angle + 360 if angle < 0 else angle

def classify_pose(landmarks, output_image):
    label, color = 'Unknown Pose', (0, 0, 255)

    if landmarks:
        left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        left_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        right_shoulder_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
        left_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        right_knee_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

        angles = {
            'left_elbow': left_elbow_angle,
            'right_elbow': right_elbow_angle,
            'left_shoulder': left_shoulder_angle,
            'right_shoulder': right_shoulder_angle,
            'left_knee': left_knee_angle,
            'right_knee': right_knee_angle
        }

        if ((150 <= left_elbow_angle <= 310 and 30 <= right_elbow_angle <= 330) and
                (0 <= left_shoulder_angle <= 50 or 290 <= left_shoulder_angle <= 360) and
                (30 <= right_shoulder_angle <= 50 or 290 <= right_shoulder_angle <= 360) and
                (150 <= left_knee_angle <= 210 and 150 <= right_knee_angle <= 210)):
            label = 'Standing'
            color = (255, 0, 0)

        if ((38 <= left_elbow_angle <= 352) and
            (54 <= right_elbow_angle <= 353) and
            (8 <= left_shoulder_angle <= 360) and
            (6 <= right_shoulder_angle <= 348) and
            (6 <= left_knee_angle <= 303) and
            (43 <= right_knee_angle <= 286)):
            label, color = 'Sitting', (255, 255, 0)

        if ((200 <= left_elbow_angle <= 280 or 50 <= left_elbow_angle <= 90) and
            (50 <= right_elbow_angle <= 110 or 160 <= right_elbow_angle <= 180) and
            (10 <= left_shoulder_angle <= 25) and
            (4 <= right_shoulder_angle <= 25) and
            (160 <= left_knee_angle <= 190) and
            (160 <= right_knee_angle <= 190)):
            label, color = 'Standing', (0, 255, 0)

    return output_image, label, angles, color




