from Idan_pro import extract_audio
import run_cordino  # הסקריפט שלך להרצת Sonic Annotator
from pathlib import Path
from extract_features import extract_features
import subprocess

def main():

    print("🎵 Extracting audio from video...")
    extract_audio()
    
    print("🎸 Running Chordino...")
    run_cordino.run_chordino()
    print("✅ Process completed.")
    print("🎶 Extracting features and aligning chords...")
    features_df = extract_features()
    print(features_df.head())
    return  features_df
