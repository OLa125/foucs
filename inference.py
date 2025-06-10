from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import tempfile
import os

app = FastAPI()

# تمكين إعدادات CORS
origins = ["*"]  # السماح لجميع النطاقات
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

EAR_THRESHOLD = 0.23

def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

@app.post("/analyze_focus/")
async def analyze_focus(video: UploadFile = File(...)):
    # حفظ الفيديو في ملف مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(await video.read())
        temp_file_path = temp_file.name

    cap = cv2.VideoCapture(temp_file_path)
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
                        focus_scores.append(100)
                    else:
                        focus_scores.append(30)
    except Exception as e:
        return {"error": f"Error processing video: {str(e)}"}
    finally:
        cap.release()
        cv2.destroyAllWindows()
        os.remove(temp_file_path)  # حذف الملف المؤقت

    if not focus_scores:
        return {"focus_score": 0}
    return {"focus_score": float(np.mean(focus_scores))}
