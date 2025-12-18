# Data loader for HuggingFace emotion datasets
# Loads pre-trained emotion datasets with zero preprocessing

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import numpy as np

def load_emotion_dataset_from_hf(dataset_name="superb/wav2vec2-base-superb-er", split="train", max_samples=None):
    """
    Load emotion dataset from HuggingFace with zero preprocessing
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split to load ("train", "validation", "test")
        max_samples: Limit number of samples (for testing)
    
    Returns:
        Dataset object with audio arrays and labels
    """
    print(f"Loading HuggingFace dataset: {dataset_name} (split: {split})...")
    
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✓ Loaded {len(dataset)} samples from HuggingFace")
    
    return dataset

def prepare_audio_for_training(dataset, target_length=48000):
    """
    Prepare audio arrays from HuggingFace dataset for training
    
    Args:
        dataset: HuggingFace dataset
        target_length: Target audio length in samples (3s at 16kHz = 48000)
    
    Returns:
        audio_arrays, labels as numpy arrays
    """
    audio_arrays = []
    labels = []
    
    print("Processing audio samples...")
    
    for idx, sample in enumerate(dataset):
        try:
            # Extract audio array
            audio_array = sample["audio"]["array"]
            
            # Extract emotion label (different datasets use different keys)
            if "emotion" in sample:
                emotion = sample["emotion"]
            elif "label" in sample:
                emotion = sample["label"]
            else:
                emotion = sample.get("labels", 0)
            
            # Normalize to fixed length
            if len(audio_array) < target_length:
                # Pad with zeros
                audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
            else:
                # Trim to target length
                audio_array = audio_array[:target_length]
            
            audio_arrays.append(audio_array)
            labels.append(emotion)
            
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1} samples...")
                
        except Exception as e:
            print(f"⚠ Error processing sample {idx}: {e}")
            continue
    
    print(f"✓ Prepared {len(audio_arrays)} audio samples")
    
    return np.array(audio_arrays), np.array(labels)

class HFEmotionDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace emotion datasets"""
    
    def __init__(self, hf_dataset, target_length=48000, augment=False):
        """
        Args:
            hf_dataset: HuggingFace dataset object
            target_length: Target audio length (samples)
            augment: Apply data augmentation
        """
        self.dataset = hf_dataset
        self.target_length = target_length
        self.augment = augment
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Extract audio
        audio = sample["audio"]["array"]
        
        # Normalize length
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)  # (1, length)
        
        # Extract label
        if "emotion" in sample:
            label = sample["emotion"]
        elif "label" in sample:
            label = sample["label"]
        else:
            label = 0
        
        label_tensor = torch.LongTensor([label])[0]
        
        # TODO: Add augmentation if needed
        
        return audio_tensor, label_tensor

# Recommended HuggingFace emotion datasets
RECOMMENDED_DATASETS = {
    "superb": "superb/wav2vec2-base-superb-er",  # IEMOCAP-based, 4 emotions
    "custom_emotion": "Rafael505c/emotion_speech",  # Custom emotions
    "clean_speech": "Matthijs/cmu-arctic-xvectors",  # Clean speech baseline
}

# Pre-trained emotion recognition models (for comparison)
PRETRAINED_EMOTION_MODELS = {
    "wav2vec2_8class": "Dpngtm/wav2vec2-emotion-recognition",  # 8 emotions (RAVDESS)
    "wav2vec2_xlsr": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",  # Multi-lingual
}

if __name__ == "__main__":
    # Test dataset loading
    print("Testing HuggingFace dataset loading...")
    
    dataset = load_emotion_dataset_from_hf(
        dataset_name="superb/wav2vec2-base-superb-er",
        split="train",
        max_samples=10
    )
    
    print(f"\nFirst sample keys: {dataset[0].keys()}")
    print(f"Audio shape: {len(dataset[0]['audio']['array'])}")
    print(f"Sample rate: {dataset[0]['audio']['sampling_rate']}")
    
    if "emotion" in dataset[0]:
        print(f"Emotion label: {dataset[0]['emotion']}")
    elif "label" in dataset[0]:
        print(f"Label: {dataset[0]['label']}")
    
    print("\n✓ Dataset loading test successful!")
