from utils.audio_processor import process_audio
from core.transcribe import transcribe_all
from core.summarizer import summarize_transcript
from core.vector_store import create_vector_store,start_chat

part_arr = process_audio("https://youtu.be/ZdG38mpAcQw?si=1veMdwNoI4fxWoYj")

print("part_Arr:\n",part_arr)

trancript = transcribe_all(part_arr)

print("Trancription:\n",trancript)

summarized = summarize_transcript(trancript)

print("Summarized:\n",summarized)

is_created = create_vector_store(summarized.get("final_summary"))

if is_created:
    start_chat(summarized.get("recommended_questions",[]))