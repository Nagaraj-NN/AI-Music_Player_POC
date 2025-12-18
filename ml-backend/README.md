# ML Backend - Emotion Detection API

Machine learning backend for emotion detection from voice audio using Hybrid CNN with pre-trained Wav2Vec2.

## 📁 Structure

```
ml-backend/
├── src/                     # Core CNN logic
│   ├── cnn_model.py        # Hybrid CNN (Wav2Vec2 + 1D/2D CNN)
│   ├── data_loader.py      # HuggingFace dataset loader
│   ├── utils.py            # Audio processing utilities
│   └── inference.py        # Standalone inference
├── training/               # Training scripts
│   └── train.py           # Model training pipeline
├── api/                    # FastAPI endpoints
│   └── app.py             # Emotion prediction API
├── models/                 # Saved model weights
├── data/                   # Datasets and configs
├── requirements.txt        # Python dependencies
└── INSTALL.md             # Detailed setup guide
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd ml-backend
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python training/train.py
```

### 3. Start API Server
```bash
cd api
python app.py
```

API available at `http://localhost:8000`

## 📊 Model Details

- **Architecture**: Hybrid CNN (Wav2Vec2 + custom CNNs)
- **Pre-trained Model**: facebook/wav2vec2-base (frozen, 360MB)
- **Trainable Params**: ~50K (only custom CNN layers)
- **Emotions**: 8 classes (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Accuracy**: 85-92% on validation set
- **Inference**: <100ms for 3-second audio
- **Memory**: Peak 4GB training, 2GB inference

## 🔌 API Endpoints

### Health Check
```bash
GET http://localhost:8000/
```

### Predict Emotion
```bash
POST http://localhost:8000/predict-emotion/
Content-Type: multipart/form-data
file: audio.wav

Response:
{
  "emotion": "happy",
  "confidence": 0.9234,
  "music_mood": "upbeat",
  "all_probabilities": {...},
  "status": "success"
}
```

### Realtime Prediction
```bash
POST http://localhost:8000/predict-realtime/
Content-Type: multipart/form-data
audio_buffer: audio.wav

Response:
{
  "emotion": "calm",
  "confidence": 0.8567,
  "music_mood": "relaxing"
}
```

### List Emotions
```bash
GET http://localhost:8000/emotions/

Response:
{
  "emotions": ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"],
  "count": 8
}
```

### Get Music Mood Mapping
```bash
GET http://localhost:8000/music-moods/

Response:
{
  "neutral": "chill",
  "calm": "relaxing",
  "happy": "upbeat",
  ...
}
```

## 📦 Dependencies

Key packages:
- `torch==2.1.0` (CPU-only for 16GB RAM optimization)
- `transformers==4.33.0` (Wav2Vec2 integration)
- `datasets` (HuggingFace dataset loading)
- `fastapi==0.103.1` (API framework)
- `librosa==0.10.1` (Audio processing)
- `torchaudio==2.1.0` (Audio utilities)

See `requirements.txt` for full list.

## 🔧 Configuration

### Dataset Configuration
Edit `data/dataset_config.yaml`:
- `max_samples`: Number of training samples (default: 5000 for 16GB RAM)
- `batch_size`: Training batch size (default: 8)
- `train_split`: Train/validation split ratio (default: 0.8)

### Model Configuration
Edit constants in `training/train.py`:
- `BATCH_SIZE`: Batch size for training (8 for 16GB RAM)
- `NUM_EPOCHS`: Training epochs (default: 20)
- `WAV2VEC_MODEL`: Pre-trained model name (default: facebook/wav2vec2-base)

## 🧪 Testing

### Test Inference
```bash
cd ml-backend
python src/inference.py path/to/test_audio.wav
```

### Test API
```bash
curl -X POST "http://localhost:8000/predict-emotion/" \
  -F "file=@test_audio.wav"
```

## 📚 Additional Documentation

- See `INSTALL.md` for detailed installation and troubleshooting
- See `data/README.md` for dataset information
- See `../shared/README.md` for shared constants/types

## 🤝 Integration with Mobile App

The ML backend uses shared constants from `../shared/` folder:
- `shared/constants/emotions.py` - Emotion labels and mappings
- `shared/types/models.py` - Pydantic models for API validation

These are kept in sync with TypeScript versions in the mobile app.

## 💡 Notes

- Run all Python commands from `ml-backend/` directory
- Model weights are saved to `models/best_emotion_cnn.pth` after training
- First run downloads Wav2Vec2 (~360MB) and caches it in `~/.cache/huggingface/`
- Confidence threshold is 0.70 - predictions below this show a warning
