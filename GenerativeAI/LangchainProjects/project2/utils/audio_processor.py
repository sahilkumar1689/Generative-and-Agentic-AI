import os
import shutil
import yt_dlp
from dotenv import load_dotenv
from pydub import AudioSegment

# Initialize the download file temp directory
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """
    Extracts audio from a YouTube video URL and saves it as a WAV file.
    """
    output_template = os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            if not info:
                raise RuntimeError(f"Failed to fetch metadata for URL: {url}")

            # Determine downloaded file path
            filename = ydl.prepare_filename(info)
            base_path, _ = os.path.splitext(filename)
            wav_filename = f"{base_path}.wav"

            if not os.path.exists(wav_filename):
                raise FileNotFoundError(f"Downloaded audio file not found at expected path: {wav_filename}")

            return wav_filename

    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Failed to download audio from YouTube URL '{url}': {e}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred during YouTube audio extraction: {e}") from e


def convert_to_wav(input_path: str, remove_source: bool = True) -> str:
    """
    Converts audio to 16kHz mono WAV format and cleans up the input file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file does not exist: {input_path}")

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    try:
        # Load and filter audio
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Export filtered audio
        audio.export(output_path, format="wav")

        # Clean up stale input file upon successful export
        if remove_source and os.path.exists(input_path) and input_path != output_path:
            os.remove(input_path)

        return output_path

    except Exception as e:
        # If conversion fails, clean up partial output if it was created
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Error converting audio file '{input_path}': {e}") from e


def partition_audio(
    input_path: str, partition_duration: int = 10, duration_name: str = "sec", remove_source: bool = True
) -> list[str]:
    """
    Splits audio into uniform duration chunks and cleans up the intermediate source file.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input audio file does not exist: {input_path}")

    try:
        audio = AudioSegment.from_wav(input_path)

        # Calculate partition window in milliseconds
        dur_name = duration_name.lower().strip()
        if dur_name in ("min", "minute", "minutes"):
            partition_ms = partition_duration * 60 * 1000
        elif dur_name in ("hr", "hour", "hours"):
            partition_ms = partition_duration * 60 * 60 * 1000
        else:
            partition_ms = partition_duration * 1000

        audio_partition_arr = []
        base_output_path = os.path.splitext(input_path)[0]

        for idx, start in enumerate(range(0, len(audio), partition_ms)):
            partitioned_audio = audio[start : start + partition_ms]
            stored_path = f"{base_output_path}_partition{idx}.wav"

            partitioned_audio.export(stored_path, format="wav")
            audio_partition_arr.append(stored_path)

        # Clean up intermediate stale file upon successful partitioning
        if remove_source and os.path.exists(input_path):
            os.remove(input_path)

        return audio_partition_arr

    except Exception as e:
        raise RuntimeError(f"Error partitioning audio file '{input_path}': {e}") from e


def process_audio(source_path: str) -> list[str]:
    """
    Full pipeline to download/load audio, process/standardize it, and partition into chunks.
    """
    wav_path = None
    filtered_path = None

    try:
        if source_path.startswith("http://") or source_path.startswith("https://"):
            print("Extracting audio from YouTube URL...")
            wav_path = download_youtube_audio(source_path)
        else:
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"Local source file not found: {source_path}")
            wav_path = source_path

        print("Filtering and standardizing audio format...")
        # If source was local, pass remove_source=False so we don't delete the user's original file
        is_temp_download = source_path.startswith(("http://", "https://"))
        filtered_path = convert_to_wav(wav_path, remove_source=is_temp_download)

        print("Partitioning audio...")
        partitioned_arr = partition_audio(filtered_path, partition_duration=10, duration_name="sec", remove_source=True)

        print(f"Successfully created {len(partitioned_arr)} partitions.")
        return partitioned_arr

    except Exception as e:
        print(f"Audio processing failed: {e}")
        # Clean up any leftover temporary files if pipeline fails midway
        for path in (wav_path, filtered_path):
            if path and os.path.exists(path) and path != source_path:
                try:
                    os.remove(path)
                except OSError:
                    pass
        raise

# print("Process: ",process_audio("https://youtu.be/ZdG38mpAcQw?si=1veMdwNoI4fxWoYj"))