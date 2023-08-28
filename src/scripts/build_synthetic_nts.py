"""Script that builds a synthetic vocie audio from reading 
the nst dataset.
TODO: Saving the dataset as a Hugging Face Dataset.
TODO: Including a dataset config file which uses the dataset stored on Hugging Face Hub in the training script.

Usage:
python src/scripts/build_synthetic_nts.py --method gtts ./data/raw_data/nst-da-train-metadata.csv ./data/raw_data/
"""

import datetime as dt
import subprocess
from pathlib import Path

import click
import pandas as pd
from gtts import gTTS
from datasets import Audio, Dataset, DatasetDict
from tqdm.auto import tqdm


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
        variant (str, optional): The eSpeak voice variant. Default is "+m1".

    Returns:
        None
    """
    subprocess.run(["espeak", "-vda", "-w", filename, variant, text])


def generate_speech_gtts(text: str, filename: Path, language="da"):
    """Generate speech from text using gTTS and save it to a file.

    Args:
        text: The text to convert to speech.
        filename: The name of the output audio file.
        language (str, optional): Language used to speek, default is 'da' (Danish).

    Returns:
        None
    """
    tts = gTTS(text, lang=language)
    tts.save(filename)


def build_huggingface_dataset(method, input_file: Path, output_dir: Path) -> DatasetDict:
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

    dataset_dict: dict[str, Dataset] = dict()
    for split in ["train", "test"]:
        # This path should be consistent with the download path
        # change values here to match syntethic dataset.
        metadata_path = input_file / Path(split) / "metadata.csv"
        metadata_df = pd.read_csv(metadata_path, low_memory=False)
        metadata_df = metadata_df[columns_to_keep.keys()]
        metadata_df = metadata_df.rename(columns=columns_to_keep)
        metadata_df.age = metadata_df.age.map(ensure_int)
        metadata_df.speaker_id = metadata_df.age.map(ensure_int)
        metadata_df.text = metadata_df.text.map(fix_text_column)
        metadata_df = metadata_df.dropna()
        metadata_df = metadata_df.convert_dtypes()
        metadata_df["dialect"] = "gtts"
        metadata_df["sex"] = "Male"  # ?
        metadata_df["speaker_id"] = 0  # ?

        # Read text with speech synthesizer and save file in correct place and format.
        for text in metadata_df.text:
            print(text)
        # think about this.
        for index, row in metadata_df.iterrows():
            if pd.isna(row["text"]):
                pass
            else:
                text_danish = row["text"]
                # where to place the file.
                filename = Path(output_dir) / row["audio"]
                # add something to metadata_df
                if method == "mac":
                    generate_speech_mac(text_danish, filename)
                elif method == "espeak":
                    generate_speech_espeak(text_danish, filename)
                elif method == "gtts":
                    generate_speech_gtts(text_danish, filename)
                else:
                    pass

        # Generate date?
        # metadata_df["recording_datetime"] = recording_datetimes
        metadata_df.to_csv(metadata_path, index=False)

        split_dataset = Dataset.from_pandas(metadata_df, preserve_index=False)
        split_dataset = split_dataset.cast_column("audio", Audio(sampling_rate=16_000))
        dataset_dict[split] = split_dataset

    return DatasetDict(dataset_dict)


@click.command()
@click.option(
    "--method",
    type=click.Choice(["mac", "espeak", "gtts"]),
    default="gtts",
    help="Choose the method for generating speech",
)
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
def main(method, input_file: Path, output_dir: Path):
    """Script that builds a synthetic voice audio from reading the nst dataset."""
    # Read the Excel file into a pandas DataFrame


if __name__ == "__main__":
    main()
