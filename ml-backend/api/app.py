# API service exposing Hybrid Emotion CNN model
# Week 2 implementation: Real-time inference with FastAPI

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchaudio
import tempfile
import os
import sys

# Add parent directory and shared folder to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'shared'))

from src.cnn_model import HybridEmotionCNN
from src.utils import load_audio, normalize_audio
from constants.emotions import EMOTION_LABELS, EMOTION_TO_MUSIC_MOOD

app = FastAPI(
    title="AI Music Mood Detection API",
    description="Detects emotion from voice frequency using Hybrid CNN",
    version="1.0.0"
)

# Global model variable
model = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    """Load the trained model on API startup"""
    global model
    
    model_path = "models/best_emotion_cnn.pth"
    wav2vec_model = "facebook/wav2vec2-base"
    
    try:
        model = HybridEmotionCNN(
            num_classes=8, 
            sample_rate=16000,
            wav2vec_model_name=wav2vec_model
        ).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"✓ Model loaded successfully on {DEVICE}")
        print(f"✓ Using pre-trained Wav2Vec2: {wav2vec_model}")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")
        print("Starting without model - train the model first!")

@app.get("/")
async def root():
    """API health check"""
    return {
        "status": "running",
        "service": "AI Music Mood Detection API",
        "model_loaded": model is not None,
        "device": DEVICE
    }

@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    """
    Predict emotion from uploaded audio file
    
    Args:
        file: Audio file (wav, mp3, etc.)
    
    Returns:
        Emotion prediction with confidence scores
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")
    
    # Validate file type
    if not file.filename.endswith(('.wav', '.mp3', '.flac', '.ogg')):
        raise HTTPException(status_code=400, detail="Invalid file format. Use .wav, .mp3, .flac, or .ogg")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Load and preprocess audio
        audio = load_audio(tmp_path, sample_rate=16000, duration=3.0)
        audio = normalize_audio(audio)
        audio = audio.unsqueeze(0).to(DEVICE)  # Add batch dimension
        
        # Run inference
        with torch.no_grad():
            outputs = model(audio)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get predicted emotion
        emotion_id = predicted.item()
        emotion = EMOTION_LABELS[emotion_id]
        confidence_score = confidence.item()
        
        # Get all emotion probabilities
        all_probabilities = {
            EMOTION_LABELS[i]: float(probabilities[0][i].item())
            for i in range(len(EMOTION_LABELS))
        }
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Apply confidence threshold (>70% as per Week 2 plan)
        if confidence_score < 0.70:
            return JSONResponse(content={
                "emotion": emotion,
                "confidence": round(confidence_score, 4),
                "music_mood": EMOTION_TO_MUSIC_MOOD[emotion],
                "all_probabilities": all_probabilities,
                "warning": "Low confidence - emotion prediction may be uncertain"
            })
        
        return JSONResponse(content={
            "emotion": emotion,
            "confidence": round(confidence_score, 4),
            "music_mood": EMOTION_TO_MUSIC_MOOD[emotion],
            "all_probabilities": all_probabilities,
            "status": "success"
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/predict-realtime/")
async def predict_realtime(audio_buffer: UploadFile = File(...)):
    """
    Real-time emotion prediction from audio buffer (Week 2)
    For use with PyAudio 3-second capture buffers
    
    Args:
        audio_buffer: Raw audio buffer (3 seconds at 16kHz)
    
    Returns:
        Quick emotion prediction (<100ms latency goal)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Similar to predict_emotion but optimized for speed
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            content = await audio_buffer.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        audio = load_audio(tmp_path, sample_rate=16000, duration=3.0)
        audio = normalize_audio(audio)
        audio = audio.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(audio)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion = EMOTION_LABELS[predicted.item()]
        
        os.unlink(tmp_path)
        
        return {
            "emotion": emotion,
            "confidence": round(confidence.item(), 4),
            "music_mood": EMOTION_TO_MUSIC_MOOD[emotion]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions/")
async def get_emotions():
    """Get list of all supported emotions"""
    return {
        "emotions": list(EMOTION_LABELS.values()),
        "count": len(EMOTION_LABELS)
    }

@app.get("/music-moods/")
async def get_music_moods():
    """Get emotion to music mood mapping"""
    return EMOTION_TO_MUSIC_MOOD

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
