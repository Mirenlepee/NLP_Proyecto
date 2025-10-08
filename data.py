#!pip install -U openai-whisper
#!pip install wget
#!pip install feedparser
#!apt update && apt install -y ffmpeg

import whisper

model = whisper.load_model("turbo")

import os

os.makedirs("ted_audios", exist_ok=True)
os.makedirs("ted_transcripts", exist_ok=True)

import feedparser
import requests
import os

rss_url = "https://feeds.feedburner.com/TEDTalks_audio"
feed = feedparser.parse(rss_url)

os.makedirs("ted_audios", exist_ok=True)

for entry in feed.entries[:300]:
    audio_url = entry.enclosures[0].href
    audio_filename = entry.title.replace("/", "_").replace("?", "_") + ".mp3"
    audio_path = os.path.join("ted_audios", audio_filename)

    if not os.path.exists(audio_path):
        try:
            r = requests.get(audio_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(audio_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Descargado: {audio_filename}")
        except requests.exceptions.RequestException as e:
            print(f"Error al descargar {audio_filename}: {e}")
    else:
        print(f"\nYa existe: {audio_filename}")

import whisper
import glob
import os

base_path = "/ted_data"
audio_path = "/ted_data/ted_audios"
transcripts_path = "/ted_data/ted_transcripts"

model = whisper.load_model("turbo")

audio_files = glob.glob("/ted_data/ted_audios/*.mp3")


for audio_file in audio_files:
    # Nombre del archivo de transcripción
    txt_filename = os.path.basename(audio_file).replace(".mp3", ".txt")
    txt_path = os.path.join(transcripts_path, txt_filename)

    # Saltar si ya existe
    if not os.path.exists(txt_path):
        print(f"Transcribiendo: {audio_file}")
        result = model.transcribe(audio_file)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(result["text"])
    else:
        print(f"Transcripción ya existe: {txt_filename}")
