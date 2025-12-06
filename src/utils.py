# Utility functions for audio processing and feature extraction

import torch
import torchaudio
import librosa
import numpy as np
from typing import Tuple

def load_audio(audio_path: str, sample_rate: int = 16000, duration: float = 3.0) -> torch.Tensor:
    """
    Load audio file and normalize to fixed duration
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (16kHz for model)
        duration: Fixed duration in seconds (3s for real-time windows)
    
    Returns:
        Audio tensor of shape (1, samples)
    """
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Normalize duration (pad or trim)
    target_length = int(sample_rate * duration)
    if waveform.shape[1] < target_length:
        # Pad with zeros
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    else:
        # Trim to target length
        waveform = waveform[:, :target_length]
    
    return waveform

def extract_mel_spectrogram(waveform: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
    """
    Extract Mel spectrogram from audio waveform
    
    Args:
        waveform: Audio tensor
        sample_rate: Sample rate
    
    Returns:
        Mel spectrogram as numpy array
    """
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    
    mel_spec = mel_transform(waveform)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    
    return mel_spec_db.numpy()

def extract_frequency_features(audio_path: str) -> dict:
    """
    Extract various frequency-domain features from audio
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Dictionary of frequency features
    """
    # Load audio with librosa
    y, sr = librosa.load(audio_path, sr=16000, duration=3.0)
    
    # Fundamental frequency (pitch)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
    
    # Zero crossing rate (measure of noisiness/percussiveness)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # Spectral centroid (brightness of sound)
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    # Spectral rolloff (measure of skewness)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # MFCCs (Mel-frequency cepstral coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    return {
        "pitch_mean": float(pitch_mean),
        "zero_crossing_rate": float(zcr),
        "spectral_centroid": float(spectral_centroid),
        "spectral_rolloff": float(spectral_rolloff),
        "mfcc_mean": mfcc_mean.tolist()
    }

def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalize audio waveform to [-1, 1] range
    
    Args:
        waveform: Audio tensor
    
    Returns:
        Normalized audio tensor
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val > 0:
        waveform = waveform / max_val
    return waveform

def apply_augmentation(waveform: torch.Tensor, augmentation_type: str = "noise") -> torch.Tensor:
    """
    Apply simple audio augmentation for training
    
    Args:
        waveform: Audio tensor
        augmentation_type: Type of augmentation ("noise", "gain", "shift")
    
    Returns:
        Augmented audio tensor
    """
    if augmentation_type == "noise":
        # Add Gaussian noise
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise
    
    elif augmentation_type == "gain":
        # Random gain adjustment
        gain = torch.FloatTensor(1).uniform_(0.8, 1.2)
        return waveform * gain
    
    elif augmentation_type == "shift":
        # Time shift
        shift = int(torch.randint(-1600, 1600, (1,)).item())  # ±0.1s at 16kHz
        return torch.roll(waveform, shift, dims=1)
    
    return waveform
