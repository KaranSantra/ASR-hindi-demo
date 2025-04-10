#!/usr/bin/env python3
# Hindi Audio Transcription Script using vasista22/whisper-hindi-large-v2

import os
import sys
import datetime
import argparse
from pathlib import Path
import torch
from transformers import pipeline


def process_hindi_audio(audio_file_path):
    """
    Process a Hindi audio file: transcribe it and save the transcription.

    This function handles the entire transcription workflow:
    1. Creates the 'transcribed' folder if it doesn't exist
    2. Loads the whisper model for Hindi
    3. Transcribes the audio file
    4. Saves the transcription with timestamp in the filename

    Args:
        audio_file_path: Path to the Hindi audio file to transcribe

    Returns:
        Path to the saved transcription file
    """
    # Check if the audio file exists
    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"The file '{audio_file_path}' does not exist.")

    # Create 'transcribed' folder if it doesn't exist
    os.makedirs("transcribed", exist_ok=True)
    print("Transcription folder 'transcribed' is ready.")

    # Set up device and model
    print(f"Loading model vasista22/whisper-hindi-large-v2...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize the transcription pipeline
    pipe = pipeline(
        task="automatic-speech-recognition",
        model="vasista22/whisper-hindi-large-v2",
        chunk_length_s=30,
        device=device,
    )

    # Transcribe the audio
    print(f"Transcribing {audio_file_path}...")
    result = pipe(audio_file_path)
    transcription = result["text"]

    # Generate timestamp and create output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    audio_filename = Path(audio_file_path).stem
    output_filename = f"{audio_filename}-{timestamp}.txt"
    output_path = os.path.join("transcribed", output_filename)

    # Save the transcription
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcription)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Hindi audio files using Whisper."
    )
    parser.add_argument("audio_file", help="Path to the Hindi audio file to transcribe")
    args = parser.parse_args()

    try:
        # Process the audio file and get the path to the saved transcription
        output_path = process_hindi_audio(args.audio_file)
        print(f"✓ Transcription saved to: {output_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
