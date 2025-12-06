# Inference utilities for the Hybrid Emotion CNN
# Standalone inference module for quick testing

import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cnn_model import HybridEmotionCNN, EMOTION_LABELS
from src.utils import load_audio, normalize_audio

class EmotionPredictor:
    """Lightweight inference class for emotion prediction using Wav2Vec2 + CNN"""
    
    def __init__(self, model_path="models/best_emotion_cnn.pth", 
                 wav2vec_model_name="facebook/wav2vec2-base", device=None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to trained model weights
            wav2vec_model_name: HuggingFace Wav2Vec2 model identifier
            device: Device to run inference on (auto-detect if None)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model with pre-trained Wav2Vec2
        self.model = HybridEmotionCNN(
            num_classes=8, 
            sample_rate=16000,
            wav2vec_model_name=wav2vec_model_name
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        print(f"✓ Emotion predictor loaded on {self.device}")
    
    def predict(self, audio_path, return_all_probs=False):
        """
        Predict emotion from audio file
        
        Args:
            audio_path: Path to audio file
            return_all_probs: Return probabilities for all emotions
        
        Returns:
            Dictionary with emotion and confidence
        """
        # Load and preprocess audio
        audio = load_audio(audio_path, sample_rate=16000, duration=3.0)
        audio = normalize_audio(audio)
        audio = audio.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(audio)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion_id = predicted.item()
        emotion = EMOTION_LABELS[emotion_id]
        confidence_score = confidence.item()
        
        result = {
            "emotion": emotion,
            "confidence": round(confidence_score, 4)
        }
        
        if return_all_probs:
            result["all_probabilities"] = {
                EMOTION_LABELS[i]: round(float(probabilities[0][i].item()), 4)
                for i in range(len(EMOTION_LABELS))
            }
        
        return result
    
    def predict_batch(self, audio_paths):
        """
        Predict emotions for multiple audio files
        
        Args:
            audio_paths: List of audio file paths
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in audio_paths:
            try:
                result = self.predict(path)
                result["file"] = path
                results.append(result)
            except Exception as e:
                results.append({
                    "file": path,
                    "error": str(e)
                })
        
        return results

def quick_predict(audio_path, model_path="models/best_emotion_cnn.pth"):
    """
    Quick single prediction without class instantiation
    
    Args:
        audio_path: Path to audio file
        model_path: Path to model weights
    
    Returns:
        Emotion string
    """
    predictor = EmotionPredictor(model_path)
    result = predictor.predict(audio_path)
    return result["emotion"]

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <audio_file>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    if not os.path.exists(audio_file):
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    predictor = EmotionPredictor()
    result = predictor.predict(audio_file, return_all_probs=True)
    
    print(f"\n🎭 Emotion Detection Result")
    print(f"{'='*40}")
    print(f"File: {audio_file}")
    print(f"Emotion: {result['emotion'].upper()}")
    print(f"Confidence: {result['confidence']*100:.2f}%")
    print(f"\nAll probabilities:")
    for emotion, prob in result['all_probabilities'].items():
        bar = '█' * int(prob * 50)
        print(f"  {emotion:12s}: {bar:50s} {prob*100:.2f}%")
