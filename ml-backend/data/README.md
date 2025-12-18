# Dataset Configuration for AI Music Mood Detection
# Optimized for 16GB RAM laptops with 5K sample training

## Recommended HuggingFace Datasets

### Primary Dataset (Week 1 Training)
- **Dataset**: `superb/wav2vec2-base-superb-er`
- **Samples**: 10K clips available (use 5K for efficient training)
- **Sample Rate**: 16kHz
- **Duration**: Variable (normalize to 3 seconds)
- **Emotions**: 4 emotions (IEMOCAP-based)
  - Neutral
  - Happy
  - Sad
  - Angry

### Alternative 8-Emotion Dataset
- **Dataset**: RAVDESS/TESS/CREMA-D combined
- **Recommended Model**: `Dpngtm/wav2vec2-emotion-recognition`
- **Emotions**: 8 classes
  - Neutral
  - Calm
  - Happy
  - Sad
  - Angry
  - Fearful
  - Disgust
  - Surprised

## Dataset Loading Configuration

### For Training (5K Samples)
```python
from datasets import load_dataset

# Stream mode for memory efficiency
dataset = load_dataset(
    "superb/wav2vec2-base-superb-er", 
    split="train",
    streaming=False  # Set to True for very large datasets
)

# Use 5K samples for 16GB RAM optimization
train_dataset = dataset.select(range(5000))
```

### Memory Optimization Strategy
- **Streaming mode**: ~500MB RAM usage
- **Batch size**: 8-16 (for 3-second audio clips)
- **Total RAM during training**: ~4GB peak
- **Free RAM remaining**: 12GB headroom

## Dataset Preparation (Zero Prep)

### Audio Normalization
- **Fixed length**: 3 seconds (48,000 samples at 16kHz)
- **Padding**: Zero-pad shorter clips
- **Trimming**: Truncate longer clips
- **No resampling needed**: HuggingFace datasets already at 16kHz

### Train/Val Split
- **Training**: 80% (~4,000 samples)
- **Validation**: 20% (~1,000 samples)
- **Stratified split**: Maintain class balance

## Sample Dataset Structure

```
data/
├── README.md                    # This file
├── dataset_config.yaml          # Dataset configuration
├── sample_audio/                # Sample audio files for testing
│   ├── neutral_sample.wav
│   ├── happy_sample.wav
│   ├── sad_sample.wav
│   └── angry_sample.wav
└── cache/                       # HuggingFace dataset cache (auto-generated)
```

## Loading Instructions

### Quick Start (Monday - Week 1)
```python
from src.data_loader import load_emotion_dataset_from_hf

# Load 5K samples
dataset = load_emotion_dataset_from_hf(
    dataset_name="superb/wav2vec2-base-superb-er",
    split="train",
    max_samples=5000
)
```

### Streaming for Large Datasets
```python
# For datasets > 10K samples
dataset = load_dataset(
    "superb/wav2vec2-base-superb-er",
    split="train",
    streaming=True  # Stream mode
)
```

## Expected Performance

### Training Metrics (5K Samples)
- **Training time**: 2-4 hours on CPU
- **RAM usage**: ~4GB peak
- **Accuracy target**: 85-92%
- **Model size**: ~15MB (custom CNN weights)

### Dataset Statistics
- **Total samples**: 5,000
- **Training samples**: 4,000
- **Validation samples**: 1,000
- **Audio duration**: 3 seconds each
- **Total audio time**: ~4.2 hours
- **Storage size**: ~1-2GB (compressed audio)

## Alternative Datasets

### 1. Custom Emotion Dataset
```python
dataset = load_dataset("Rafael505c/emotion_speech")
```

### 2. Clean Speech (Baseline)
```python
dataset = load_dataset("Matthijs/cmu-arctic-xvectors")
```

### 3. Pre-trained Emotion Model
```python
from transformers import pipeline

# Use pre-trained instead of training from scratch
classifier = pipeline(
    "audio-classification",
    model="Dpngtm/wav2vec2-emotion-recognition"
)
```

## Notes

- All datasets are loaded directly from HuggingFace (zero manual download)
- Datasets are cached locally after first download
- Streaming mode available for memory-constrained environments
- 5K samples provide good balance between accuracy and training time
- Can scale to 10K samples if more RAM available

## Citation

If using IEMOCAP-based dataset:
```
@inproceedings{busso2008iemocap,
  title={IEMOCAP: Interactive emotional dyadic motion capture database},
  author={Busso, Carlos and Bulut, Murtaza and Lee, Chi-Chun and others},
  booktitle={Language resources and evaluation},
  year={2008}
}
```
