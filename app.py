from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from inference import analyze_focus  # لازم تكون دي دالة موجودة في inference.py
import shutil
import os

app = FastAPI(title="Focus Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze_focus")
async def analyze_focus_api(video: UploadFile = File(...)):
    temp_path = f"temp_{video.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    try:
        focus_score = analyze_focus(temp_path)  # بنفترض دي الدالة الأساسية
        return {"focus_score": f"{focus_score:.2f}%"}
    finally:
        os.remove(temp_path)