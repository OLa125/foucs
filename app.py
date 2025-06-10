from fastapi import FastAPI, UploadFile, File
from inference import analyze_focus
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import tempfile

app = FastAPI()

# تمكين إعدادات CORS
origins = ["*"]  # السماح لجميع النطاقات. يمكنك تحديد نطاقات معينة إذا أردت.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # السماح للنطاقات المحددة فقط
    allow_credentials=True,
    allow_methods=["*"],  # السماح بجميع الطرق (GET, POST, ...)
    allow_headers=["*"],  # السماح بجميع الرؤوس
)

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    # حفظ الفيديو في ملف مؤقت
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        shutil.copyfileobj(file.file, temp_file)
        temp_file_path = temp_file.name

    try:
        # استدعاء الدالة لتحليل الفيديو
        score = analyze_focus(temp_file_path)
    except Exception as e:
        return {"error": f"Error analyzing video: {str(e)}"}
    finally:
        # حذف الملف المؤقت
        os.remove(temp_file_path)

    return {"focus_score": score}

