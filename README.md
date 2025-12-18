# AI Music Mood Detection - Monorepo

A cross-platform music recommendation app that detects emotions from voice using AI and suggests music via Spotify API.

## 🎯 Project Overview

This monorepo contains:
- **ML Backend**: Hybrid CNN (Wav2Vec2 + custom CNNs) for emotion detection from audio
- **Mobile App**: React Native app for iOS, Android, and Web (coming soon)
- **Shared Libraries**: Type-safe constants and API contracts shared between backend and frontend

### Key Features
- **Emotion Detection**: 8-class emotion recognition (neutral, calm, happy, sad, angry, fearful, disgust, surprised)
- **Lightweight ML**: Optimized for 16GB RAM laptops with <100ms inference
- **Cross-Platform**: Single codebase for mobile (React Native) and web
- **Type-Safe Integration**: Shared TypeScript/Python types ensure consistency
- **Spotify Integration**: Maps emotions to music moods for personalized recommendations

## 📁 Monorepo Structure

```
AI-Music_Player_POC/
├── ml-backend/                   # Python ML backend (FastAPI)
│   ├── src/                     # Core CNN logic
│   │   ├── cnn_model.py         # Hybrid CNN (Wav2Vec2 + 1D/2D CNN)
│   │   ├── data_loader.py       # HuggingFace dataset loader
│   │   ├── utils.py             # Audio processing utilities
│   │   └── inference.py         # Standalone inference
│   ├── training/                # Model training scripts
│   │   └── train.py             # Training pipeline with Wav2Vec2
│   ├── api/                     # FastAPI endpoints
│   │   └── app.py               # Emotion prediction API
│   ├── models/                  # Saved model weights
│   ├── data/                    # Datasets and configs
│   ├── requirements.txt         # Python dependencies
│   └── INSTALL.md              # ML backend setup guide
│
├── mobile-app/                   # React Native app (iOS/Android/Web)
│   └── .gitkeep                 # Placeholder (app coming soon)
│
├── shared/                       # Shared code between backend and frontend
│   ├── constants/               # Shared constants
│   │   ├── emotions.py          # Python version (ML backend)
│   │   └── emotions.ts          # TypeScript version (mobile app)
│   ├── types/                   # Type definitions
│   │   ├── models.py            # Python Pydantic models
│   │   └── api-contracts.ts     # TypeScript interfaces
│   └── README.md                # Shared libraries documentation
│
├── docs/                         # Project documentation
│   └── 3-Week Plan...           # Development roadmap
│
└── README.md                     # This file
```

## 🏗️ ML Architecture

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
- **ML Backend**: Python 3.8+, 16GB RAM, CUDA GPU (optional)
- **Mobile App**: Node.js 18+, React Native CLI (coming soon)

### ML Backend Setup

1. Clone the repository:
```bash
git clone https://github.com/Nagaraj-NN/AI-Music_Player_POC.git
cd AI-Music_Player_POC
```

2. Install Python dependencies:
```bash
cd ml-backend
pip install -r requirements.txt
```

3. Train the model:
```bash
python training/train.py
```

**Training outputs:**
- `models/best_emotion_cnn.pth` - Best model weights (~200KB)
- `models/emotion_cnn.onnx` - ONNX export for production

**Expected accuracy:** 85-92% on validation set

### Running the ML API

#### Option 1: Standalone Inference
```bash
cd ml-backend
python src/inference.py path/to/audio.wav
```

#### Option 2: API Service
Start the FastAPI server:
```bash
cd ml-backend/api
python app.py
```

API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /` - Health check
- `POST /predict-emotion/` - Upload audio file for emotion prediction
- `POST /predict-realtime/` - Real-time prediction from audio buffer
- `GET /emotions/` - List supported emotions
- `GET /music-moods/` - Get emotion-to-mood mapping

### Mobile App Setup (Coming Soon)

```bash
cd mobile-app
npm install
npm run start     # Start Metro bundler
npm run android   # Run on Android
npm run ios       # Run on iOS
npm run web       # Run web version
```

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
