"""Script that builds a synthetic vocie audio from reading 
the nst dataset.

Usage:
python src/scripts/build_synthetic_nts.py --method gtts ./data/nst/
"""
import time
import os
import datetime as dt
import subprocess
from pathlib import Path
import requests.exceptions

import click
import pandas as pd
from gtts import gTTS
from datasets import Audio, Dataset, DatasetDict
from pydub import AudioSegment
from tqdm.auto import tqdm


def get_audio_file_info(filename):
    """
    Get information about an audio file.

    Args:
        filename (str): The path to the audio file.

    Returns:
        dict: A dictionary containing audio file information.
            - 'duration_ms' (int): Duration in milliseconds.
            - 'duration_seconds' (float): Duration in seconds.
            - 'channels' (int): Number of audio channels (mono or stereo).
            - 'sample_width_bytes' (int): Sample width in bytes.
            - 'frame_rate' (int): Frame rate in samples per second.

    Example:
        audio_info = get_audio_file_info("your_audio_file.mp3")
        print(audio_info)
    """
    # Load the audio file
    audio = AudioSegment.from_file(filename)

    # Get the duration in milliseconds
    duration_ms = len(audio)

    # Convert duration to seconds
    duration_seconds = duration_ms / 1000

    # Get the number of channels (mono or stereo)
    channels = audio.channels

    # Get the sample width (in bytes)
    sample_width_bytes = audio.sample_width

    # Get the frame rate (samples per second)
    frame_rate = audio.frame_rate

    # Create a dictionary with the extracted information
    audio_info = {
        'duration_ms': duration_ms,
        'duration_seconds': duration_seconds,
        'channels': channels,
        'sample_width_bytes': sample_width_bytes,
        'frame_rate': frame_rate
    }

    return audio_info


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


def generate_speech_espeak(text: str, filename: Path, variant="+m1"):
    """Generate speech from text using eSpeak and save it to a file.

    Args:
        text (str): The text to convert to speech.
        filename (str): The name of the output audio file.
        variant (str): The eSpeak voice variant. Default is "+m1".

    Returns:
        None
    """
    subprocess.run(["espeak", "-vda", "-w", filename, variant, text])


def generate_speech_gtts(text: str, filename: str, language="da"):
    """Generate speech from text using gTTS and save it to a file.

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.
        language (str, optional): Language used to speek, default is 'da' (Danish).

    Returns:
        None
    """
    tts = gTTS(text, lang=language)
    if os.path.exists(filename):
        print(f"The file {filename} exists.")
    else:
        print(f"The file {filename} does not exist.")
        tts.save(filename)


class GTTSRateLimitError(Exception):
    pass


def generate_speech_gtts_with_retry(text: str, filename: str, language="dk", max_retries=300):
    """Generate speech from text using gTTS with retry handling and save it to a file.

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.
        language (str, optional): Language used to speak, default is 'en' (English).
        max_retries (int, optional): Maximum number of retries for HTTP 429 errors.

    Returns:
        None
    """
    retries = 0
    while retries < max_retries:
        try:
            # Create the gTTS object
            # audio = gTTS(text=text, lang=language)
            # Save the audio file
            # audio.save(filename)
            print(filename)
            print(text)
            generate_speech_gtts(text, filename)
            # print("Waiting for 1 seconds... Rate limit exists.")
            # time.sleep(1)
            retries = 0
            break  # Break out of the loop if successful
        except Exception as e:
            if "429 (Too Many Requests)" or "Max retries" in str(e):
                if retries == 0:
                    retries += 1
                    print("Rate limited. Waiting for 10 seconds...")
                    time.sleep(10)
                if retries == 1:
                    retries += 1
                    print("Rate limited. Waiting for 100 seconds...")
                    time.sleep(100)
                if retries == 2:
                    retries += 1
                    print("Rate limited. Waiting for 200 seconds...")
                    time.sleep(200)
                if retries == 3:
                    retries += 1
                    print("Rate limited. Waiting for 1 hour...")
                    time.sleep(60*60)
                if retries == 4:
                    retries += 1
                    print("Rate limited. Waiting for 4 hours...")
                    time.sleep(4*60*60)
                if retries > 4:
                    retries += 1
                    print("Rate limited. Waiting for 1 day...")
                    time.sleep(60*60*24)
            else:
                raise e
    else:
        raise Exception("Maximum retries exceeded for generating speech")


def build_synthetic_dataset(method, input_file: Path = Path('./data/nst/')) -> DatasetDict:
    """Sets up the metadata files and builds the Hugging Face dataset.

    Returns:
        The Hugging Face dataset.
    """

    def ensure_int(value: int | str | None) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def fix_text_column(text: str) -> str:
        if text == "( ... tavshed under denne indspilning ...)":
            return ""
        text = text.replace("\Komma", "").replace("\Punktum", "")
        return text

    columns_to_keep = {
        "filename_both_channels": "audio",
        "text": "text",
        "Speaker_ID": "speaker_id",
        "Age": "age",
        "Sex": "sex",
        "Region_of_Dialect": "dialect",
        "RecDate": "recording_date",
        "RecTime": "recording_time",
    }

    # dataset_dict: dict[str, Dataset] = dict()
    for split in ["test", "train"]:
        # This path should be consistent with the download path
        # change values here to match syntethic dataset.
        metadata_path = input_file / Path(split) / "metadata.csv"
        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        metadata_df = metadata_df.rename(columns=columns_to_keep)
        metadata_df.age = metadata_df.age.map(ensure_int)
        metadata_df.speaker_id = metadata_df.age.map(ensure_int)
        metadata_df.text = metadata_df.text.map(fix_text_column)
        # metadata_df = metadata_df.dropna()
        metadata_df = metadata_df.convert_dtypes()
        # Read text with speech synthesizer and save file in correct place and format.
        # for text in metadata_df.text:
        #    print(text)
        # think about this.
        for index, row in tqdm(
            iterable=metadata_df[::-1].iterrows(),
            total=len(metadata_df),
            desc=f"Extracting file names for the {split} split",
        ):
            if pd.isna(row["text"]):
                metadata_df = metadata_df.drop(index=metadata_df.iloc[index].name)
                pass
            else:
                text_danish = fix_text_column(row["text"])
                # where to place the file.
                filename = Path(input_file) / row["audio"]
                # Data to metafile.
                metadata_df.iloc[index, metadata_df.columns.get_loc("speaker_id")] = 0
                metadata_df.iloc[index, metadata_df.columns.get_loc(
                    "recording_datetime")] = dt.datetime.now().isoformat()
                if method == "mac":
                    generate_speech_mac(text_danish, filename)
                    metadata_df.iloc[index,
                                     metadata_df.columns.get_loc("dialect")] = "mac"

                elif method == "espeak":
                    generate_speech_espeak(text_danish, filename)
                    metadata_df.iloc[index, metadata_df.columns.get_loc(
                        "dialect")] = "espeak"

                elif method == "gtts":
                    generate_speech_gtts_with_retry(text_danish, filename)
                    # generate_speech_gtts(text_danish, filename)
                    metadata_df.iloc[index, metadata_df.columns.get_loc(
                        "dialect")] = "gtts"


def build_synthetic_huggingface_dataset():
    '''
    TODO:
    audio_info = get_audio_file_info(filename)
    metadata_df.to_csv(metadata_path, index=False)

    split_dataset = Dataset.from_pandas(metadata_df, preserve_index=False)
    split_dataset = split_dataset.cast_column(
        "audio", Audio(sampling_rate=audio_info["frame_rate"]))
    dataset_dict[split] = split_dataset

    return DatasetDict(dataset_dict)
    '''


@click.command()
@click.option(
    "--method",
    type=click.Choice(["mac", "espeak", "gtts"]),
    default="gtts",
    help="Choose the method for generating speech",
)
@click.argument("input_file", type=click.Path(exists=True))
# @click.argument("output_dir", type=click.Path())
def main(method, input_file: Path):
    """Script that builds a synthetic voice audio from reading the nst dataset."""
    input_file_path = Path(input_file)

    build_synthetic_dataset(method, input_file_path)
    # hugging_face =


if __name__ == "__main__":
    main()
