# Hybrid CNN model using pre-trained Wav2Vec2 as feature extractor
# Architecture: Wav2Vec2 (frozen) → Custom 1D-CNN + 2D-CNN → Emotion Classification

import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class CNN1D(nn.Module):
    """1D CNN for raw audio waveform processing"""
    def __init__(self):
        super(CNN1D, self).__init__()
        # Process raw audio waveform (temporal features)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=80, stride=4)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(4)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, 1, audio_length)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        return x

class CNN2D(nn.Module):
    """2D CNN for Mel spectrogram processing"""
    def __init__(self):
        super(CNN2D, self).__init__()
        # Process Mel spectrogram (time-frequency features)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2, 2))
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2, 2))
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d((2, 2))
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x shape: (batch, 1, n_mels, time)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        return x

class HybridEmotionCNN(nn.Module):
    """Hybrid CNN using pre-trained Wav2Vec2 features + Custom 1D/2D CNN classifiers"""
    def __init__(self, num_classes=8, sample_rate=16000, wav2vec_model_name="facebook/wav2vec2-base"):
        super(HybridEmotionCNN, self).__init__()
        
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        
        # Pre-trained Wav2Vec2 feature extractor (FROZEN)
        print(f"Loading pre-trained Wav2Vec2: {wav2vec_model_name}...")
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(wav2vec_model_name)
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec_model_name)
        
        # Freeze Wav2Vec2 parameters (use as feature extractor only)
        for param in self.wav2vec2.parameters():
            param.requires_grad = False
        
        print("✓ Wav2Vec2 loaded and frozen (feature extractor mode)")
        
        # Wav2Vec2 output dimension is 768 for base model
        wav2vec_dim = 768
        
        # Mel spectrogram transform for 2D CNN branch
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        
        # 1D CNN branch (processes Wav2Vec2 features)
        # Input: (batch, wav2vec_dim, time_steps)
        self.cnn1d_wav2vec = nn.Sequential(
            nn.Conv1d(wav2vec_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)  # Global pooling
        )
        
        # 2D CNN branch (processes Mel spectrogram)
        self.cnn2d = CNN2D()
        
        # Feature fusion and classification
        # 64 from 1D-CNN + flattened 2D-CNN features
        self.fc1 = nn.Linear(64 + 128 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """
        Args:
            x: Raw audio waveform (batch, 1, audio_length)
            Expected audio_length: ~48000 for 3-second clips at 16kHz
        """
        batch_size = x.size(0)
        
        # Remove channel dimension for Wav2Vec2
        x_flat = x.squeeze(1)  # (batch, audio_length)
        
        # Extract Wav2Vec2 features (frozen, no gradients)
        with torch.no_grad():
            wav2vec_output = self.wav2vec2(x_flat).last_hidden_state
            # Output shape: (batch, time_steps, 768)
        
        # 1D CNN path on Wav2Vec2 features
        # Transpose to (batch, 768, time_steps) for Conv1d
        wav2vec_features = wav2vec_output.transpose(1, 2)
        features_1d = self.cnn1d_wav2vec(wav2vec_features)
        features_1d = features_1d.view(batch_size, -1)  # Flatten
        
        # 2D CNN path on Mel spectrogram
        mel_spec = self.mel_transform(x)
        mel_spec = mel_spec.unsqueeze(1)  # Add channel dimension
        features_2d = self.cnn2d(mel_spec)
        features_2d = features_2d.view(batch_size, -1)  # Flatten
        
        # Fuse features from both branches
        combined = torch.cat([features_1d, features_2d], dim=1)
        
        # Classification head
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Emotion labels (8-class system based on RAVDESS/TESS)
EMOTION_LABELS = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}
