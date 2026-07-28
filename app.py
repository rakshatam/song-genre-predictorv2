from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets
import uvicorn
import shutil
import os
import torch
import librosa
from collections import Counter
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from moviepy import VideoFileClip
import logging

logging.getLogger("moviepy").setLevel(logging.ERROR)

app = FastAPI(title="Song Genre Prediction API", version="1.0")

security = HTTPBasic()

def get_current_username(credentials: HTTPBasicCredentials = Depends(security)):
    # You can change this username and password later!
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, "xxxxxxxx")
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# Critical for Hybrid Architecture: Allows your Cloudflare frontend to talk to local API!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
feature_extractor = None
model = None
# Forcing CPU to test deployment performance as requested
device = torch.device("cpu")

@app.on_event("startup")
def load_model():
    global feature_extractor, model
    print(f"Starting up. Using device: {device}")
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    
    local_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    print(f"Loading AST model {model_name} into memory from {local_cache_dir}...")
    
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name, cache_dir=local_cache_dir)
    model = ASTForAudioClassification.from_pretrained(model_name, cache_dir=local_cache_dir).to(device)
    model.eval()
    print("Model loaded successfully.")

def extract_audio(input_path, output_audio_path="temp_audio.wav"):
    ext = os.path.splitext(input_path)[1].lower()
    if ext in ['.mp4', '.avi', '.mkv', '.mov']:
        video = VideoFileClip(input_path)
        audio = video.audio
        audio.write_audiofile(output_audio_path, logger=None)
        audio.close()
        video.close()
        return output_audio_path
    return input_path

def predict_audio_chunk(chunk, sr):
    """Helper function to run inference on a single 10.24s chunk for file uploads"""
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
            
    return tags

@app.post("/predict")
async def predict_genre(file: UploadFile = File(...), username: str = Depends(get_current_username)):
    """The original full-file endpoint for accurate predictions"""
    if model is None or feature_extractor is None:
        raise HTTPException(status_code=503, detail="Model is still loading...")
    
    temp_input_path = f"temp_{file.filename}"
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        audio_path = extract_audio(temp_input_path, "temp_audio.wav")
        
        target_sr = 16000
        y, sr = librosa.load(audio_path, sr=target_sr)
        chunk_samples = int(10.24 * sr)
        
        all_predictions = []
        # Chunking loop
        for i in range(0, len(y), chunk_samples):
            chunk = y[i:i + chunk_samples]
            if len(chunk) > 3 * sr:
                try:
                    tags = predict_audio_chunk(chunk, sr)
                    all_predictions.extend(tags)
                except Exception as e:
                    print(f"Chunk error: {e}")
                
        if not all_predictions:
            raise HTTPException(status_code=400, detail="Could not process audio.")
            
        counter = Counter(all_predictions)
        
        return {
            "filename": file.filename,
            "top_genres": {genre: f"detected in {count} chunks" for genre, count in counter.most_common(5)},
        }
        
    finally:
        if os.path.exists(temp_input_path):
            try: os.remove(temp_input_path)
            except: pass
        if os.path.exists("temp_audio.wav"):
            try: os.remove("temp_audio.wav")
            except: pass

if __name__ == "__main__":
    uvicorn.run("app:app", host="x.x.x.x", port=xxxx, reload=False)
