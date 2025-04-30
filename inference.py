import cv2
import numpy as np
import mediapipe as mp
import csv
import time

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]
LEFT_IRIS = [468]
RIGHT_IRIS = [473]

EAR_THRESHOLD = 0.23

def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def get_head_pose(face_landmarks, iw, ih):
    face_2d = []
    face_3d = []

    for lm in face_landmarks.landmark:
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

def calculate_focus_score(focus_status, avg_ear, gaze):
    if focus_status == "Focused":
        if gaze == "Center" and avg_ear > EAR_THRESHOLD:
            return 100
        elif gaze != "Center" and avg_ear > EAR_THRESHOLD:
            return 80
    elif focus_status == "Not Focused":
        if avg_ear <= EAR_THRESHOLD:
            return 30
        elif gaze == "Center":
            return 60
        else:
            return 50
    return 0

def analyze_focus(video_path):
    cap = cv2.VideoCapture(video_path)
    focus_scores = []
    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ih, iw, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_results = face_detection.process(frame_rgb)
        mesh_results = face_mesh.process(frame_rgb)

        if detection_results.detections and mesh_results.multi_face_landmarks:
            for detection, face_landmarks in zip(detection_results.detections, mesh_results.multi_face_landmarks):
                pitch, yaw, roll, focus_status = get_head_pose(face_landmarks, iw, ih)
                if pitch is None:
                    continue

                left_eye = np.array([[face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih] for i in [33, 160, 158, 133, 153, 144]])
                right_eye = np.array([[face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih] for i in [362, 385, 387, 263, 373, 380]])
                avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                lcx = int(face_landmarks.landmark[LEFT_IRIS[0]].x * iw)
                rcx = int(face_landmarks.landmark[RIGHT_IRIS[0]].x * iw)
                lx1, lx2 = int(face_landmarks.landmark[LEFT_EYE[0]].x * iw), int(face_landmarks.landmark[LEFT_EYE[1]].x * iw)
                rx1, rx2 = int(face_landmarks.landmark[RIGHT_EYE[0]].x * iw), int(face_landmarks.landmark[RIGHT_EYE[1]].x * iw)

                left_ratio = (lcx - lx1) / (lx2 - lx1 + 1e-6)
                right_ratio = (rcx - rx2) / (rx1 - rx2 + 1e-6)
                avg_ratio = (left_ratio + (1 - right_ratio)) / 2

                if avg_ratio < 0.35:
                    gaze = "Looking Right"
                elif avg_ratio > 0.65:
                    gaze = "Looking Left"
                else:
                    gaze = "Center"

                score = calculate_focus_score(focus_status, avg_ear, gaze)
                focus_scores.append(score)
                frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    if not focus_scores:
        return 0
    return float(np.mean(focus_scores))
