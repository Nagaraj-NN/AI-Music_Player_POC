# Training script for Hybrid Emotion CNN using pre-trained Wav2Vec2 + Custom CNNs
# Week 1 implementation: Wav2Vec2 (frozen) → 1D-CNN + 2D-CNN → Emotion classifier

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cnn_model import HybridEmotionCNN, EMOTION_LABELS
from src.utils import normalize_audio, apply_augmentation

class EmotionDataset(Dataset):
    """Custom Dataset for emotion audio from HuggingFace"""
    def __init__(self, audio_arrays, labels, sample_rate=16000, augment=False):
        self.audio_arrays = audio_arrays
        self.labels = labels
        self.sample_rate = sample_rate
        self.augment = augment
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Convert to tensor and ensure correct shape
        audio = torch.FloatTensor(self.audio_arrays[idx])
        
        # Ensure mono and add channel dimension
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        
        # Normalize audio
        audio = normalize_audio(audio)
        
        # Apply augmentation during training
        if self.augment and np.random.random() > 0.5:
            aug_type = np.random.choice(["noise", "gain", "shift"])
            audio = apply_augmentation(audio, aug_type)
        
        label = torch.LongTensor([self.labels[idx]])[0]
        
        return audio, label

def load_hf_emotion_dataset(dataset_name="superb/wav2vec2-base-superb-er", max_samples=None):
    """
    Load emotion dataset from HuggingFace
    
    Args:
        dataset_name: HF dataset identifier
        max_samples: Limit number of samples (for testing)
    
    Returns:
        audio_arrays, labels
    """
    print(f"Loading dataset: {dataset_name}...")
    dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    audio_arrays = []
    labels = []
    
    for sample in dataset:
        try:
            audio_array = sample["audio"]["array"]
            emotion = sample["emotion"] if "emotion" in sample else sample["label"]
            
            # Ensure fixed length (3 seconds at 16kHz = 48000 samples)
            target_length = 48000
            if len(audio_array) < target_length:
                audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
            else:
                audio_array = audio_array[:target_length]
            
            audio_arrays.append(audio_array)
            labels.append(emotion)
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    print(f"Loaded {len(audio_arrays)} samples")
    return np.array(audio_arrays), np.array(labels)

def train_model(model, train_loader, val_loader, num_epochs=20, device="cuda"):
    """
    Train the Hybrid Emotion CNN
    
    Args:
        model: HybridEmotionCNN instance
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of training epochs
        device: Training device
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, (audio, labels) in enumerate(train_loader):
            audio, labels = audio.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_acc = accuracy_score(train_labels, train_preds)
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for audio, labels in val_loader:
                audio, labels = audio.to(device), labels.to(device)
                outputs = model(audio)
                _, predicted = torch.max(outputs, 1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Acc: {val_acc:.4f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "models/best_emotion_cnn.pth")
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")
        
        scheduler.step(val_acc)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.4f}")
    print("\nFinal validation report:")
    print(classification_report(val_labels, val_preds, target_names=[EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]))

def export_to_onnx(model, device="cuda"):
    """
    Export trained model to ONNX format for production (Week 1 Friday)
    
    Args:
        model: Trained HybridEmotionCNN
        device: Device to use
    """
    model.eval()
    dummy_input = torch.randn(1, 1, 48000).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        "models/emotion_cnn.onnx",
        export_params=True,
        opset_version=11,
        input_names=["audio"],
        output_names=["emotion"],
        dynamic_axes={"audio": {0: "batch_size"}, "emotion": {0: "batch_size"}}
    )
    
    print("✓ Model exported to models/emotion_cnn.onnx")

def main():
    # Configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8  # Reduced to 8 for 16GB RAM (was 16)
    NUM_EPOCHS = 20
    NUM_CLASSES = 8  # 8-emotion system
    WAV2VEC_MODEL = "facebook/wav2vec2-base"  # Pre-trained model
    
    print(f"Using device: {DEVICE}")
    print(f"Wav2Vec2 model: {WAV2VEC_MODEL}")
    print(f"Batch size: {BATCH_SIZE} (optimized for 16GB RAM)")
    
    # Load dataset from HuggingFace
    audio_arrays, labels = load_hf_emotion_dataset(
        dataset_name="superb/wav2vec2-base-superb-er",
        max_samples=5000  # Optimized for 16GB RAM (remove limit for full training)
    )
    
    # Split into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        audio_arrays, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create datasets
    train_dataset = EmotionDataset(X_train, y_train, augment=True)
    val_dataset = EmotionDataset(X_val, y_val, augment=False)
    
    # Create dataloaders with reduced num_workers for 16GB RAM
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model with pre-trained Wav2Vec2
    model = HybridEmotionCNN(
        num_classes=NUM_CLASSES, 
        sample_rate=16000,
        wav2vec_model_name=WAV2VEC_MODEL
    ).to(DEVICE)
    
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, device=DEVICE)
    
    # Load best model and export to ONNX
    model.load_state_dict(torch.load("models/best_emotion_cnn.pth"))
    export_to_onnx(model, device=DEVICE)
    
    print("\n🎉 Training pipeline complete!")

if __name__ == "__main__":
    main()
