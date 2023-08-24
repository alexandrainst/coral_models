"""Script that builds a synthetic vocie audio from reading 
the nst dataset. 

Usage:
    python build_syntethic_nts.py 
"""

import os
import subprocess

from gtts import gTTS
import pandas as pd
from pathlib import Path



def generate_speech_mac(text: str, filename: Path):
    """Generate speech from text using macOS 'say' command and save it to a file.

    Note, these voice has a licens only for private use.
    The downloaded voices needs to be changed manually in 
    spoken content in settings of the mac machine.
    The file is saved in a special mac sound format called ".aiff".
    Extra details see: 
    "https://maithegeek.medium.com/having-fun-in-macos-with-say-command-d4a0d3319668"

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.

    Returns:
        None
    """
    subprocess.run(["say", text, "-o", filename])

def generate_speech_eSpeak(text: str, filename: Path, variant="+m1"):
    """Generate speech from text using eSpeak and save it to a file.

    Args:
        text (str): The text to convert to speech.
        filename (str): The name of the output audio file.
        variant (str, optional): The eSpeak voice variant. Default is "+m1".

    Returns:
        None
    """
    subprocess.run(["espeak", "-vda", "-w", filename, variant, text])


def generate_speech_gTTS(text: str, filename: Path, language="da"):
    """Generate speech from text using gTTS and save it to a file.

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.
        language (str, optional): The language to use for the speech. Default is 'da' (Danish).

    Returns:
        None
    """
    tts = gTTS(text, lang=language)
    tts.save(filename)


def main():
    """Main function to generate speech from text using gTTS."""

    # Read nst data.
    csv_file = "./data/raw_data/nst-da-train-metadata.csv"

    # Read the Excel file into a pandas DataFrame
    columns = ["audio", "text", "speaker_id", "age", "sex", "dialect", "recording_datetime"]
    df = pd.read_csv(csv_file, usecols=columns)

    # folder for saving files.
    folder = "./data/raw_data/"

    for index, row in df.iterrows():
        if pd.isna(row["text"]):
            pass
        else:
            # Get text and file name
            text_danish = row["text"]
            filename = os.path.join(folder, row["audio"])
            generate_speech_from_gTTS(text=text_danish, filename=filename)

if __name__ == "__main__":
    main()
