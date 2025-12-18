"""
Shared emotion constants for ML backend and mobile app
Keep in sync with emotions.ts
"""

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

# Emotion to music mood mapping (for Spotify integration)
EMOTION_TO_MUSIC_MOOD = {
    "neutral": "chill",
    "calm": "relaxing",
    "happy": "upbeat",
    "sad": "uplifting",
    "angry": "energetic",
    "fearful": "calming",
    "disgust": "neutral",
    "surprised": "exciting"
}

# Confidence thresholds
CONFIDENCE_THRESHOLD = {
    "HIGH": 0.85,
    "MEDIUM": 0.70,
    "LOW": 0.50
}

# Emotion colors for UI (hex codes)
EMOTION_COLORS = {
    "neutral": "#94A3B8",
    "calm": "#60A5FA",
    "happy": "#FBBF24",
    "sad": "#3B82F6",
    "angry": "#EF4444",
    "fearful": "#8B5CF6",
    "disgust": "#10B981",
    "surprised": "#F97316"
}
