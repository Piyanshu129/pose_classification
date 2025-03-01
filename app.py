# from collections import deque
# import numpy as np
# from flask import Flask, render_template, request, Response, jsonify
# import cv2
# import pose_detection
#
# app = Flask(__name__)
#
# pose = pose_detection.init_pose()
# angle_buffer = deque(maxlen=10)
# frame_skip_interval = 5
# frame_count = 0
#
# def generate_frames(rtsp_url):
#     global frame_count
#     cap = cv2.VideoCapture(rtsp_url)
#     if not cap.isOpened():
#         print("Error: Unable to open RTSP stream")
#         return
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Unable to read frame from RTSP stream")
#             break
#
#         frame_count += 1
#
#         if frame_count % frame_skip_interval == 0:
#             frame_resized = cv2.resize(frame, (640, 480))
#             output_image, landmarks, bbox = pose_detection.detect_pose(frame_resized, pose)
#             label, angles, color = 'Unknown Pose', {}, (0, 0, 255)
#
#             if landmarks:
#                 output_image, label, angles, color = pose_detection.classify_pose(landmarks, output_image)
#                 angle_buffer.append(angles)
#
#                 if len(angle_buffer) == angle_buffer.maxlen:
#                     left_shoulder_var = np.var([a['left_shoulder'] for a in angle_buffer])
#                     right_shoulder_var = np.var([a['right_shoulder'] for a in angle_buffer])
#                     if all(2 <= var <= 10 for var in [left_shoulder_var, right_shoulder_var]):
#                         label = 'Sitting'
#                         color = (255, 255, 0)
#
#                 cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
#                 if bbox:
#                     cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
#
#                 # Store detected angles and bounding box coordinates in global variables
#                 global detected_angles, bbox_coordinates
#                 detected_angles = angles
#                 bbox_coordinates = bbox
#
#             ret, buffer = cv2.imencode('.jpg', output_image)
#             if not ret:
#                 continue
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#
#     cap.release()
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
# @app.route('/video_feed')
# def video_feed():
#     rtsp_url = request.args.get('rtsp_url', default='', type=str)
#     return Response(generate_frames(rtsp_url),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')
#
# @app.route('/pose_info')
# def pose_info():
#     # Return detected angles and bbox coordinates as JSON
#     return jsonify({
#         'angles': detected_angles,
#         'bbox': bbox_coordinates
#     })
#
# if __name__ == '__main__':
#     # Initialize global variables
#     detected_angles = {}
#     bbox_coordinates = None
#     app.run(debug=True)
#
#
#
#
#
#
#
#
#
#
#





from collections import deque
import numpy as np
from flask import Flask, render_template, request, Response, jsonify
import cv2
import pose_detection
import os
import pose_detection

app = Flask(__name__)

# Directory to save detected pose images
IMAGE_DIR = 'static/images'
os.makedirs(IMAGE_DIR, exist_ok=True)
pose = pose_detection.init_pose()
angle_buffer = deque(maxlen=10)
frame_skip_interval = 5
frame_count = 0
def generate_frames(rtsp_url):
    global frame_count
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from RTSP stream")
            break

        frame_count += 1

        if frame_count % frame_skip_interval == 0:
            frame_resized = cv2.resize(frame, (640, 480))
            output_image, landmarks, bbox = pose_detection.detect_pose(frame_resized, pose)
            label, angles, color = 'Unknown Pose', {}, (0, 0, 255)

            if landmarks:
                output_image, label, angles, color = pose_detection.classify_pose(landmarks, output_image)
                angle_buffer.append(angles)

                if len(angle_buffer) == angle_buffer.maxlen:
                    left_shoulder_var = np.var([a['left_shoulder'] for a in angle_buffer])
                    right_shoulder_var = np.var([a['right_shoulder'] for a in angle_buffer])
                    if all(2 <= var <= 10 for var in [left_shoulder_var, right_shoulder_var]):
                        label = 'Sitting'
                        color = (255, 255, 0)

                cv2.putText(output_image, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                if bbox:
                    cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                # Store detected angles and bounding box coordinates in global variables
                global detected_angles, bbox_coordinates
                detected_angles = angles
                bbox_coordinates = bbox

                # Save the detected pose image
                image_path = os.path.join(IMAGE_DIR, 'detected_pose.jpg')
                cv2.imwrite(image_path, output_image)

            ret, buffer = cv2.imencode('.jpg', output_image)
            if not ret:
                continue
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    rtsp_url = request.args.get('rtsp_url', default='', type=str)
    return Response(generate_frames(rtsp_url),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_info')
def pose_info():
    # Return detected angles and bbox coordinates as JSON
    return jsonify({
        'angles': detected_angles,
        'bbox': bbox_coordinates
    })

@app.route('/pose_image')
def pose_image():
    return Response(open(os.path.join(IMAGE_DIR, 'detected_pose.jpg'), 'rb').read(), mimetype='image/jpeg')

if __name__ == '__main__':
    # Initialize global variables
    detected_angles = {}
    bbox_coordinates = None
    app.run(debug=True)

