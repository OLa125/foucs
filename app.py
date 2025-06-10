from fastapi import FastAPI, UploadFile, File
from inference import analyze_focus
import shutil
import os

app = FastAPI()

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    with open("temp_video.mp4", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    score = analyze_focus("temp_video.mp4")
    os.remove("temp_video.mp4")

    return {"focus_score": score}
