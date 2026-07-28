import os
import argparse
from collections import Counter
import torch
import librosa
from transformers import ASTFeatureExtractor, ASTForAudioClassification

# We use moviepy to extract audio from video files (mp4, mkv, etc.)
from moviepy import VideoFileClip
import logging
logging.getLogger("moviepy").setLevel(logging.ERROR)

def extract_audio(input_path, output_audio_path="temp_audio.wav"):
    """Extracts audio from video or returns the audio file path."""
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mkv', '.mov']:
        print(f"Extracting audio from video file {input_path}...")
        video = VideoFileClip(input_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path, logger=None)
        audio.close()
        video.close()
        return output_audio_path
    else:
        return input_path

def chunk_audio(audio_path, chunk_duration=10.24):
    """Loads audio and splits it into chunks.
    AST model was trained on 10.24-second clips (1024 frames).
    """
    print(f"Loading audio {audio_path}...")
    target_sr = 16000
    y, sr = librosa.load(audio_path, sr=target_sr)
    
    # 10.24 seconds * 16000 sr = 163840 samples
    chunk_samples = int(chunk_duration * sr)
    chunks = []
    
    for i in range(0, len(y), chunk_samples):
        chunk = y[i:i + chunk_samples]
        if len(chunk) > 3 * sr:
            chunks.append(chunk)
            
    return chunks, target_sr

def predict_genre(input_file):
    # Forcing CPU to test deployment performance as requested
    device = torch.device("cpu")
    print(f"Using device: {device} (Forced for testing)")

    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    local_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    print(f"Loading AST model {model_name} (527 classes)...")
    print(f"Model cache directory: {local_cache_dir}")
    
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name, cache_dir=local_cache_dir)
    model = ASTForAudioClassification.from_pretrained(model_name, cache_dir=local_cache_dir).to(device)
    model.eval()

    audio_path = extract_audio(input_file)
    
    chunks, sr = chunk_audio(audio_path, chunk_duration=10.24)
    
    print(f"Divided into {len(chunks)} chunks of ~10.24 seconds.")
    
    all_predictions = []
    
    for i, chunk in enumerate(chunks):
        print(f"\nProcessing chunk {i+1}/{len(chunks)}...")
        try:
            # Manual extraction ensures proper padding (max_length) and normalization for AST
            inputs = feature_extractor(chunk, sampling_rate=sr, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0]
            
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            
            tags = []
            for idx_tensor in sorted_indices:
                idx = idx_tensor.item()
                # AudioSet Ontology ranges:
                # 27 to 37: Vocal Music (Singing, Choir, Rapping, etc.)
                # 137 to 282: Music Genres, Instruments, Moods, Soundtracks
                if (27 <= idx <= 37) or (137 <= idx <= 282):
                    if idx == 137: # Redundant root node "Music"
                        continue
                    tag = model.config.id2label[idx]
                    tags.append(tag)
                if len(tags) >= 3:
                    break
            
            all_predictions.extend(tags)
            print(f"Top tags for chunk {i+1}: {tags}")
        except Exception as e:
            print(f"Error processing chunk {i+1}: {e}")

    if not all_predictions:
        print("No predictions could be made.")
        return None

    print("\n===============================")
    print("        FINAL ANALYSIS         ")
    print("===============================")
    
    counter = Counter(all_predictions)
    print("Most prominent genres across the entire song:")
    for genre, count in counter.most_common(5):
        print(f" - {genre} (detected in {count} chunks)")
    
    if audio_path == "temp_audio.wav" and os.path.exists(audio_path):
        os.remove(audio_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict song genre from audio/video using AST.")
    parser.add_argument("input_file", help="Path to the input media file (.mp3, .mp4, etc.)")
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"Error: File not found: {args.input_file}")
    else:
        predict_genre(args.input_file)
