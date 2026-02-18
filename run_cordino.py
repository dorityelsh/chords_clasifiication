import subprocess
from pathlib import Path
import pandas as pd
from io import StringIO


def run_chordino():
    PROJECT_ROOT = Path(__file__).parent

    SONIC_ANNOTATOR = PROJECT_ROOT / "sonic-annotator-folder" / "sonic-annotator"
    AUDIO_FILE = PROJECT_ROOT / "output" / "audio.wav"
    CSV_OUTPUT = PROJECT_ROOT / "output" / "chords.csv"

    # בדיקות בסיס
    if not SONIC_ANNOTATOR.exists():
        raise FileNotFoundError(f"sonic-annotator not found: {SONIC_ANNOTATOR}")

    if not AUDIO_FILE.exists():
        raise FileNotFoundError(f"audio file not found: {AUDIO_FILE}")

 
    cmd = [
        str(SONIC_ANNOTATOR),
        "-d", "vamp:nnls-chroma:chordino:simplechord",
        "-w", "csv",
        "--csv-stdout",
        str(AUDIO_FILE)
    ]


    print("🎸 Running chord extraction...")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    df = pd.read_csv(StringIO(result.stdout))

   
    df = df.iloc[:, 1:3]
    df.columns = ["timestamp", "label"]


    df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Cleaned Chords CSV saved at: {CSV_OUTPUT}")

    print(f"✅ Chords CSV created at: {CSV_OUTPUT}")
