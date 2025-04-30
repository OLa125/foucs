!pip install mediapipe





import cv2
import numpy as np
import mediapipe as mp
import csv
import time

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

# Models
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Landmark indices
LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468]
RIGHT_IRIS = [473]

# EAR function
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Head Pose Estimation
def get_head_pose(face_landmarks, iw, ih):
    face_2d = []
    face_3d = []

    for idx, lm in enumerate(face_landmarks.landmark):
        x, y = int(lm.x * iw), int(lm.y * ih)
        face_2d.append([x, y])
        face_3d.append([x, y, lm.z])

    face_2d = np.array(face_2d, dtype=np.float64)
    face_3d = np.array(face_3d, dtype=np.float64)

    focal_length = iw
    cam_matrix = np.array([[focal_length, 0, iw / 2],
                           [0, focal_length, ih / 2],
                           [0, 0, 1]], dtype=np.float64)

    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

    if not success:
        return None, None, None, None

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0] * 360
    yaw = angles[1] * 360
    roll = angles[2] * 360

    focus_status = "Focused" if -13 < yaw < 13 and -25 < pitch < 25 else "Not Focused"

    return pitch, yaw, roll, focus_status

# Load video
video_path = 'AMR.mp4'  # Change if needed
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width, frame_height)

output_path = 'output_final.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

EAR_THRESHOLD = 0.23

# CSV file setup
csv_file = 'focus_scores.csv'
csv_header = ['Frame', 'Focus_Score', 'Timestamp']
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(csv_header)

# Define function to calculate focus score
def calculate_focus_score(focus_status, avg_ear, gaze):
    if focus_status == "Focused":
        if gaze == "Center" and avg_ear > EAR_THRESHOLD:
            return 100  # Full focus
        elif gaze != "Center" and avg_ear > EAR_THRESHOLD:
            return 80  # Partial focus with gaze off-center
    elif focus_status == "Not Focused":
        if avg_ear <= EAR_THRESHOLD:  # Eyes closed
            return 30  # Low focus with eyes closed
        elif gaze == "Center":
            return 60  # Lower focus with eyes open and centered gaze
        else:
            return 50  # Lower focus with eyes open but not centered gaze
    return 0  # Default to 0 if no conditions match

# Initialize variables for averaging
focus_scores = []
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("✅ End of video.")
        break

    ih, iw, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(frame_rgb)
    mesh_results = face_mesh.process(frame_rgb)

    if detection_results.detections and mesh_results.multi_face_landmarks:
        for detection, face_landmarks in zip(detection_results.detections, mesh_results.multi_face_landmarks):
            # Bounding Box
            bboxC = detection.location_data.relative_bounding_box
            x1, y1 = int(bboxC.xmin * iw), int(bboxC.ymin * ih)
            w, h = int(bboxC.width * iw), int(bboxC.height * ih)

            # Head Pose
            pitch, yaw, roll, focus_status = get_head_pose(face_landmarks, iw, ih)

            # Draw face box
            color = (0, 255, 0) if focus_status == "Focused" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)

            # Display Pose
            if pitch is not None:
                cv2.putText(frame, f"{focus_status}", (x1, y1 - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"Pitch: {pitch:.2f}", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Yaw: {yaw:.2f}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Roll: {roll:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # EAR (Eye Open/Closed)
            left_eye_landmarks = np.array([[
                face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih
                ] for i in [33, 160, 158, 133, 153, 144]
            ])
            right_eye_landmarks = np.array([[
                face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih
                ] for i in [362, 385, 387, 263, 373, 380]
            ])

            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)
            avg_ear = (left_ear + right_ear) / 2
            eye_status = "Open" if avg_ear > EAR_THRESHOLD else "Closed"

            # Iris and Gaze
            lcx = int(face_landmarks.landmark[LEFT_IRIS[0]].x * iw)
            lcy = int(face_landmarks.landmark[LEFT_IRIS[0]].y * ih)
            rcx = int(face_landmarks.landmark[RIGHT_IRIS[0]].x * iw)
            rcy = int(face_landmarks.landmark[RIGHT_IRIS[0]].y * ih)

            lx1 = int(face_landmarks.landmark[LEFT_EYE[0]].x * iw)
            lx2 = int(face_landmarks.landmark[LEFT_EYE[1]].x * iw)
            rx1 = int(face_landmarks.landmark[RIGHT_EYE[0]].x * iw)
            rx2 = int(face_landmarks.landmark[RIGHT_EYE[1]].x * iw)

            left_ratio = (lcx - lx1) / (lx2 - lx1 + 1e-6)
            right_ratio = (rcx - rx2) / (rx1 - rx2 + 1e-6)
            avg_ratio = (left_ratio + (1 - right_ratio)) / 2

            if avg_ratio < 0.35:
                gaze = "Looking Right"
            elif avg_ratio > 0.65:
                gaze = "Looking Left"
            else:
                gaze = "Center"

            # Calculate Focus Score
            focus_score = calculate_focus_score(focus_status, avg_ear, gaze)
            timestamp = time.time() - start_time
            focus_scores.append(focus_score)
            frame_count += 1

            # Log Focus Score to CSV
            with open(csv_file, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_count, focus_score, timestamp])

    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Average Focus Score per Second
with open(csv_file, mode='r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header
    frame_focus_scores = {}
    for row in reader:
        timestamp = float(row[2])
        second = int(timestamp)
        focus_score = float(row[1])
        if second not in frame_focus_scores:
            frame_focus_scores[second] = []
        frame_focus_scores[second].append(focus_score)

    avg_focus_scores = {second: np.mean(scores) for second, scores in frame_focus_scores.items()}

    # Output average scores per second and determine focus status
    overall_focus_score = np.mean(list(avg_focus_scores.values()))  # Overall average focus score
    print(f"Overall Average Focus Score = {overall_focus_score:.2f}")

    for second, avg_score in avg_focus_scores.items():
        focus_status = "Focused" if avg_score >= 50 else "Not Focused"
        print(f"Second {second}: Average Focus Score = {avg_score:.2f} - {focus_status}")


