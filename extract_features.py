import pandas as pd
import numpy as np
import librosa
from pathlib import Path

# ---------------------------
# הגדרות קבצים
# ---------------------------
PROJECT_ROOT = Path(__file__).parent
AUDIO_FILE = PROJECT_ROOT / "output" / "audio.wav"
CHORDINO_CSV = PROJECT_ROOT / "output" / "chords.csv"

# ---------------------------
# פונקציית חישוב פיצ'רים
# ---------------------------
def extract_features(audio_path=AUDIO_FILE, chords_csv=CHORDINO_CSV, save_csv=True):
    """
    טוען קובץ שמע, מחשב פיצ'רים (Chroma + MFCC) ומצמיד אקורדים.
    
    :param audio_path: Path לקובץ WAV
    :param chords_csv: Path ל־Chordino CSV
    :param save_csv: האם לשמור את הפיצ'רים ל־CSV
    :return: DataFrame עם פיצ'רים ואקורדים
    """
    print("Audio path:", audio_path)
    print("Exists?", audio_path.exists())

    # 1️⃣ טעינת שמע
    y, sr = librosa.load(str(audio_path), sr=None)
    print(f"Loaded audio: {len(y)/sr:.2f} seconds at {sr}Hz")
    print("Librosa version:", librosa.__version__)
    print("Numpy version:", np.__version__)

    try:
        # חישוב ספקטרוגרמה
        S = np.abs(librosa.stft(y))**2
        print("Spectrogram OK")
        print("sr:", sr)

        # חישוב Chroma
    
        chroma = librosa.feature.chroma_stft(S=S, sr=sr)
        

    except Exception as e:
        print("Error computing Chroma:", e)
        raise

    # חישוב Chroma + MFCC ל־DataFrame
    chroma_df = pd.DataFrame(chroma.T, columns=[f"chroma_{i}" for i in range(12)])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_df = pd.DataFrame(mfcc.T, columns=[f"mfcc_{i}" for i in range(13)])

    features_df = pd.concat([chroma_df, mfcc_df], axis=1)
    features_df['time'] = librosa.frames_to_time(np.arange(features_df.shape[0]), sr=sr)

    # 2️⃣ טען תוויות (Chordino)
    chords_df = pd.read_csv(chords_csv)

    def assign_chords(features_times, chord_times, chord_labels):
        labels = []
        chord_idx = 0
        for t in features_times:
            while chord_idx + 1 < len(chord_times) and t > chord_times[chord_idx + 1]:
                chord_idx += 1
            labels.append(chord_labels[chord_idx])
        return labels

    features_df['chord'] = assign_chords(
        features_df['time'].values,
        chords_df['timestamp'].values,
        chords_df['label'].values
    )

    # 3️⃣ שמירה והחזרה
    if save_csv:
        output_path = Path(audio_path.parent) / "features_with_chords.csv"
        features_df.to_csv(output_path, index=False)
        print(f"✅ Features + Chords CSV saved at {output_path}")

    print(f"DataFrame shape: {features_df.shape}")
    return features_df



