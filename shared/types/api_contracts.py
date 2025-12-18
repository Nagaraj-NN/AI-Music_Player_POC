"""
Corresponding Python Pydantic models
Keep in sync with TypeScript definitions in api-contracts.ts
"""
from pydantic import BaseModel, Field
from typing import Literal, Dict, Optional
from enum import Enum

# ===== EMOTION TYPES =====
class EmotionType(str, Enum):
    NEUTRAL = "neutral"
    CALM = "calm"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    DISGUST = "disgust"
    SURPRISED = "surprised"

class MusicMoodType(str, Enum):
    CHILL = "chill"
    RELAXING = "relaxing"
    UPBEAT = "upbeat"
    UPLIFTING = "uplifting"
    ENERGETIC = "energetic"
    CALMING = "calming"
    NEUTRAL = "neutral"
    EXCITING = "exciting"

# ===== RESPONSE MODELS =====
class EmotionPredictionResponse(BaseModel):
    emotion: EmotionType
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    music_mood: MusicMoodType
    all_probabilities: Dict[str, float]
    status: Literal["success", "low_confidence"] = "success"
    warning: Optional[str] = None

class RealtimePredictionResponse(BaseModel):
    emotion: EmotionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    music_mood: MusicMoodType

class APIHealthResponse(BaseModel):
    status: Literal["running", "error"]
    service: str
    model_loaded: bool
    device: Literal["cuda", "cpu"]

class APIError(BaseModel):
    detail: str
    status_code: int
    timestamp: Optional[str] = None

# ===== SPOTIFY INTEGRATION =====
class SpotifyQueryParams(BaseModel):
    seed_genres: Optional[str] = None
    target_valence: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_energy: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_acousticness: Optional[float] = Field(None, ge=0.0, le=1.0)
    target_tempo: Optional[str] = None
    target_loudness: Optional[str] = None
    limit: int = Field(20, ge=1, le=100)
    market: str = "US"
