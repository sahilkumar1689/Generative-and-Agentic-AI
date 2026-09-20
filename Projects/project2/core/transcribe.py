import os
import whisper

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

_model = None


def load_whisper_model():
    """
    Loads and caches the Whisper model in memory.
    """
    global _model

    if _model is None:
        print(f"Loading Whisper model '{WHISPER_MODEL}'...")
        try:
            _model = whisper.load_model(WHISPER_MODEL)
            print("Whisper model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model '{WHISPER_MODEL}': {e}") from e

    return _model


def transcribe_partition(partition_path: str, translate: bool = False, remove_partition: bool = True) -> str:
    """
    Transcribes a single audio partition file using Whisper and cleans up the file afterward.
    """
    if not os.path.exists(partition_path):
        raise FileNotFoundError(f"Partition audio file not found: {partition_path}")

    try:
        model = load_whisper_model()
        task = "translate" if translate else "transcribe"

        result = model.transcribe(partition_path, task=task)
        result_text = result.get("text", "").strip()

        if not result_text:
            result_text = "[No text extracted]"

        print(f"Result for '{os.path.basename(partition_path)}':\n{result_text}\n")
        return result_text

    except Exception as e:
        print(f"Error transcribing '{partition_path}': {e}")
        raise RuntimeError(f"Transcription failed for partition '{partition_path}': {e}") from e

    finally:
        # Cleanup the partition file after transcription (or failure)
        if remove_partition and os.path.exists(partition_path):
            try:
                os.remove(partition_path)
            except OSError as cleanup_err:
                print(f"Warning: Could not remove partition file '{partition_path}': {cleanup_err}")


def transcribe_all(partition_arr: list[str], translate: bool = False, remove_partitions: bool = True) -> str:
    """
    Transcribes a list of audio partitions sequentially and joins their transcripts into a single text.
    """
    if not partition_arr:
        print("Warning: Received an empty partition list for transcription.")
        return ""

    transcript_parts = []

    for idx, partition in enumerate(partition_arr):
        print(f"Processing partition {idx + 1} of {len(partition_arr)}: {os.path.basename(partition)}")
        try:
            text = transcribe_partition(partition, translate=translate, remove_partition=remove_partitions)
            if text and text != "[No text extracted]":
                transcript_parts.append(text)
        except Exception as e:
            print(f"Skipping partition {idx + 1} due to error: {e}")

    full_transcript = " ".join(transcript_parts)
    print("Transcription Completed.")
    return full_transcript