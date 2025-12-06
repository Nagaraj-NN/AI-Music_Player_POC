# Installation guide for 16GB RAM laptops
# Optimized for CPU-only training

## Quick Install (Recommended)

### For CPU-only training (saves 2-3GB RAM):
```bash
# Install PyTorch CPU version first
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### For GPU training (if you have NVIDIA GPU):
```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

## Memory Footprint

### Installed Packages (~3-4GB total):
- PyTorch CPU: ~200MB
- TorchAudio: ~50MB
- Transformers: ~1.5GB
- Datasets: ~300MB
- Librosa: ~200MB
- FastAPI/Uvicorn: ~70MB
- NumPy/Scikit-learn: ~300MB
- ONNX Runtime: ~200MB
- Other utilities: ~200MB

### Runtime Memory Usage (16GB RAM):
- **Training peak**: ~4GB
  - PyTorch runtime: ~1.5GB
  - Wav2Vec2 model: ~1.5GB (loaded once, frozen)
  - Data batch (8 samples): ~500MB
  - Gradients + optimizer: ~500MB
- **Free RAM**: ~12GB headroom
- **Inference**: ~2GB (Wav2Vec2 + CNN)

## Verification

After installation, verify memory-efficient setup:

```python
import torch
import sys

# Check PyTorch version and device
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CPU cores: {torch.get_num_threads()}")

# Estimate model size
from transformers import Wav2Vec2Model
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
param_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Wav2Vec2 size: {param_size / 1024**2:.2f} MB")
```

## Troubleshooting

### Out of Memory during training?
1. Reduce batch size to 4 in `training/train.py`
2. Reduce max_samples to 3000
3. Enable gradient checkpointing (saves ~30% memory)

### Slow download of Wav2Vec2?
- Model is cached in `~/.cache/huggingface/`
- First download: ~360MB (one-time)
- Subsequent runs: loaded from cache

### Cannot install PyAudio?
- PyAudio is removed from requirements (not needed for file-based training)
- For real-time audio (Week 2), we'll use sounddevice instead

## Minimal Install (Training only)

If you only want to train the model without API:

```bash
pip install torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets librosa soundfile numpy scikit-learn
```

This reduces installation size to ~2.5GB.
