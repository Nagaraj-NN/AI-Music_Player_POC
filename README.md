# AI Music Mood Detection POC

An AI-powered music recommendation app that detects emotions from voice frequency using a lightweight Hybrid CNN and suggests music based on detected mood.

## 🎯 Project Overview

This project uses a **Hybrid CNN** (1D-CNN + 2D-CNN) to detect emotions from voice audio and maps them to appropriate music moods for personalized music recommendations via Spotify.

### Key Features
- **Emotion Detection**: 8-class emotion recognition (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Hybrid CNN Architecture**: Combines raw waveform (1D-CNN) and Mel spectrogram (2D-CNN) features
- **Lightweight**: Optimized for 16GB RAM laptops
- **Real-time**: <100ms inference latency for 3-second audio clips
- **API-First**: FastAPI service for easy integration with apps
- **HuggingFace Integration**: Zero data prep using pre-existing emotion datasets

## 📁 Project Structure

```
AI-Music_Player_POC/
├── src/                          # Core CNN logic (separate from training)
│   ├── __init__.py              # Package initialization
│   ├── cnn_model.py             # Hybrid CNN (Wav2Vec2 + 1D/2D CNN)
│   ├── data_loader.py           # HuggingFace dataset loader (zero prep)
│   ├── utils.py                 # Audio processing utilities
│   └── inference.py             # Standalone inference module
│
├── training/                     # Training scripts (isolated from core logic)
│   ├── __init__.py              # Package initialization
│   └── train.py                 # Model training pipeline with Wav2Vec2
│
├── api/                          # API service layer (exposes model to app)
│   ├── __init__.py              # Package initialization
│   └── app.py                   # FastAPI endpoints for emotion prediction
│
├── models/                       # Saved model weights (generated after training)
│   ├── best_emotion_cnn.pth     # Trained CNN weights (only custom layers)
│   └── emotion_cnn.onnx         # ONNX production model
│
├── data/                         # Sample audio data and datasets
│   ├── README.md                # Dataset documentation and loading guide
│   ├── dataset_config.yaml      # Dataset configuration (5K samples)
│   └── cache/                   # HuggingFace dataset cache (auto-generated)
│
├── requirements.txt              # Python dependencies (optimized for 16GB RAM)
└── README.md                    # This file
```

## 🏗️ Architecture

### Hybrid CNN with Pre-trained Wav2Vec2
The model leverages **pre-trained Wav2Vec2** as a frozen feature extractor and adds custom CNN classifiers:

**Architecture Flow:**
```
Raw Audio (16kHz, 3s)
    ↓
┌─────────────────────────────────────────────┐
│  Pre-trained Wav2Vec2 (FROZEN)              │
│  - Trained on 960h LibriSpeech             │
│  - 768-dim contextual speech features       │
└─────────────────────────────────────────────┘
    ↓
┌──────────────────┬──────────────────────────┐
│  1D-CNN Branch   │    2D-CNN Branch         │
│  (Wav2Vec2       │    (Mel Spectrogram)     │
│   features)      │                          │
│  - Conv1D layers │    - Conv2D layers       │
│  - 128→64 dims   │    - 32→64→128 filters   │
└──────────────────┴──────────────────────────┘
    ↓                      ↓
    └──────────┬───────────┘
               ↓
    Feature Fusion (Concat)
               ↓
    Dense Layers (256→128→8)
               ↓
    8 Emotion Classes
```

### Why Wav2Vec2 + Custom CNN?

1. **Pre-trained Wav2Vec2** (Frozen)
   - Provides rich speech representations learned from 960 hours of audio
   - Captures phonetic, prosodic, and acoustic features
   - No training required - used as feature extractor only
   - Output: 768-dimensional contextual features

2. **1D-CNN Branch** (Trainable)
   - Processes Wav2Vec2 features temporally
   - Learns emotion-specific patterns from pre-trained representations
   - Lightweight: Only ~50K parameters to train

3. **2D-CNN Branch** (Trainable)
   - Processes Mel spectrogram for complementary frequency features
   - Captures harmonic and spectral patterns
   - Multi-layer convolution for spatial patterns

4. **Feature Fusion**
   - Concatenates features from both branches
   - Dense layers for final emotion classification
   - Dropout for regularization

### Emotion → Music Mood Mapping
- **Neutral/Calm** → Chill/Relaxing playlists
- **Happy** → Upbeat music
- **Sad/Fearful** → Uplifting/Calming music
- **Angry** → Energetic music
- **Surprised** → Exciting music

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- 16GB RAM (minimum)
- CUDA-capable GPU (optional, for faster training)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Nagaraj-NN/AI-Music_Player_POC.git
cd AI-Music_Player_POC
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training the Model

Train the Hybrid CNN on HuggingFace emotion datasets:

```bash
python training/train.py
```

**Training outputs:**
- `models/best_emotion_cnn.pth` - Best model weights
- `models/emotion_cnn.onnx` - ONNX export for production

**Expected accuracy:** 85-92% on validation set (with full dataset)

### Running Inference

#### Option 1: Standalone Inference
```bash
python src/inference.py path/to/audio.wav
```

#### Option 2: API Service
Start the FastAPI server:
```bash
cd api
python app.py
```

API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /` - Health check
- `POST /predict-emotion/` - Upload audio file for emotion prediction
- `POST /predict-realtime/` - Real-time prediction from audio buffer
- `GET /emotions/` - List supported emotions
- `GET /music-moods/` - Get emotion-to-mood mapping

### Example API Usage

```python
import requests

# Upload audio file
with open("voice_sample.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict-emotion/",
        files={"file": f}
    )

result = response.json()
print(f"Emotion: {result['emotion']}")
print(f"Confidence: {result['confidence']}")
print(f"Music Mood: {result['music_mood']}")
```

## 📊 Model Performance

| Metric | Target | Notes |
|--------|--------|-------|
| Accuracy | 85-92% | Wav2Vec2 features boost performance |
| Inference Latency | <100ms | Frozen Wav2Vec2 + lightweight CNN |
| Model Size | ~15MB | Only custom CNN weights saved |
| Wav2Vec2 Size | ~360MB | Downloaded once, cached by HuggingFace |
| RAM Usage | <4GB | Optimized for 16GB laptops |
| Training Time | 2-4 hours | On 5000 samples with CPU/GPU |

## 🛠️ Development Roadmap

### ✅ Week 1: CNN Development with Wav2Vec2
- [x] HuggingFace dataset loader (zero prep)
- [x] Pre-trained Wav2Vec2 integration (frozen)
- [x] 1D-CNN on Wav2Vec2 features
- [x] 2D-CNN on Mel spectrograms
- [x] Hybrid CNN fusion architecture
- [x] Model training pipeline
- [x] ONNX export capability

### 📋 Week 2: Real-time Pipeline
- [ ] PyAudio integration for mic capture
- [ ] 3-second audio buffering
- [ ] Real-time inference (<100ms)
- [ ] Confidence filtering (>70%)
- [ ] Streamlit UI prototype

### 📋 Week 3: Spotify Integration
- [ ] Spotify API integration
- [ ] Emotion-to-playlist mapping
- [ ] Auto-play functionality
- [ ] Docker deployment
- [ ] Final demo and documentation

## 🔧 Configuration

### Model Hyperparameters
- **Sample Rate**: 16kHz
- **Audio Duration**: 3 seconds
- **Batch Size**: 32
- **Learning Rate**: 0.001
- **Epochs**: 20
- **Optimizer**: Adam
- **Scheduler**: ReduceLROnPlateau

### Audio Processing
- **Mel Spectrogram**: 64 mel bins, 1024 FFT, 512 hop length
- **Normalization**: [-1, 1] range
- **Augmentation**: Noise, gain, time shift

## 📦 Dependencies

Key packages (see `requirements.txt` for full list):
- `torch` - Deep learning framework
- `torchaudio` - Audio processing
- `transformers` - HuggingFace Wav2Vec2 models
- `datasets` - HuggingFace datasets (zero prep)
- `librosa` - Audio feature extraction
- `fastapi` - API framework
- `uvicorn` - ASGI server
- `spotipy` - Spotify API client (Week 3)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is for POC/educational purposes.

## 🙏 Acknowledgments

- HuggingFace for emotion recognition datasets
- PyTorch team for the framework
- FastAPI for the excellent API framework

## 📧 Contact

**Nagaraj Nune**
- GitHub: [@Nagaraj-NN](https://github.com/Nagaraj-NN)

---

**Built with ❤️ for AI-powered music experiences**
