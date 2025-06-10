import cv2
import numpy as np
import mediapipe as mp
import os

# إعداد Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

EAR_THRESHOLD = 0.23  # قيمة العتبة لمعدل نسبة فتح العين

def eye_aspect_ratio(eye_landmarks):
    """
    حساب معدل نسبة فتح العين (EAR).
    """
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

def analyze_focus(video_path):
    """
    تحليل التركيز من خلال فيديو مسجل.
    """
    cap = cv2.VideoCapture(video_path)
    focus_scores = []

    try:
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
                    left_eye = np.array([[face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih] for i in [33, 160, 158, 133, 153, 144]])
                    right_eye = np.array([[face_landmarks.landmark[i].x * iw, face_landmarks.landmark[i].y * ih] for i in [362, 385, 387, 263, 373, 380]])
                    avg_ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

                    if avg_ear > EAR_THRESHOLD:
                        focus_scores.append(100)  # حالة التركيز
                    else:
                        focus_scores.append(30)  # حالة عدم التركيز
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not focus_scores:
        return 0
    return float(np.mean(focus_scores))
