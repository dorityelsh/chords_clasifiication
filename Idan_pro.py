# extract_audio.py
import ffmpeg
from pathlib import Path


INPUT_FILE = Path("input/acho.m4a")  # או video.mp4
OUTPUT_FILE = Path("output/audio.wav")

# יצירת תיקיית פלט אם אין
OUTPUT_FILE.parent.mkdir(exist_ok=True)

def extract_audio():
    try:
        ffmpeg.input(str(INPUT_FILE)).output(
            str(OUTPUT_FILE),
            ac=1,
            ar=44100,
            format="wav"
        ).overwrite_output().run(quiet=True)
        print(f"✅ Audio extracted successfully to: {OUTPUT_FILE}")
    except Exception as e:
        print("❌ Error extracting audio:", e)
        raise e 


