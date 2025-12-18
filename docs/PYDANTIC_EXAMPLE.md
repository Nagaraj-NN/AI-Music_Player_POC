# How to Use Pydantic Models in ML Backend

## Current Situation
Your `shared/types/models.py` exists but isn't being used in `ml-backend/api/app.py`.

## What You're Missing

### 1. Automatic Validation
```python
# Without Pydantic (current)
return {"confidence": 1.5}  # Bug! Should be 0-1, but no error ❌

# With Pydantic
return EmotionPredictionResponse(
    emotion="happy",
    confidence=1.5,  # ValidationError: ensure this value is less than or equal to 1 ✅
    music_mood="upbeat",
    all_probabilities={...}
)
```

### 2. Auto-Generated API Documentation
FastAPI uses Pydantic models to generate interactive docs at `/docs`:
- Shows exact response structure
- Includes field descriptions and constraints
- Provides example responses

### 3. Type Safety in IDE
```python
response = EmotionPredictionResponse(...)
response.emotion      # IDE autocomplete ✅
response.emtion       # IDE error: attribute doesn't exist ✅
response.confidence = "high"  # IDE error: must be float ✅
```

## One-Shot Example: Update Your API

### Step 1: Update imports in `ml-backend/api/app.py`

```python
# Add after existing imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

from types.models import (
    EmotionPredictionResponse, 
    RealtimePredictionResponse,
    APIHealthResponse
)
```

### Step 2: Update `/` endpoint

**Before:**
```python
@app.get("/")
async def root():
    return {
        "status": "running",
        "service": "AI Music Mood Detection API",
        "model_loaded": model is not None,
        "device": DEVICE
    }
```

**After:**
```python
@app.get("/", response_model=APIHealthResponse)
async def root():
    return APIHealthResponse(
        status="running",
        service="AI Music Mood Detection API",
        model_loaded=model is not None,
        device=DEVICE
    )
```

### Step 3: Update `/predict-emotion/` endpoint

**Before (current):**
```python
@app.post("/predict-emotion/")
async def predict_emotion(file: UploadFile = File(...)):
    # ... processing ...
    
    if confidence_score < 0.70:
        return JSONResponse(content={
            "emotion": emotion,
            "confidence": round(confidence_score, 4),
            "music_mood": EMOTION_TO_MUSIC_MOOD[emotion],
            "all_probabilities": all_probabilities,
            "warning": "Low confidence - emotion prediction may be uncertain"
        })
    
    return JSONResponse(content={
        "emotion": emotion,
        "confidence": round(confidence_score, 4),
        "music_mood": EMOTION_TO_MUSIC_MOOD[emotion],
        "all_probabilities": all_probabilities,
        "status": "success"
    })
```

**After (with Pydantic):**
```python
@app.post("/predict-emotion/", response_model=EmotionPredictionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    # ... same processing code ...
    
    return EmotionPredictionResponse(
        emotion=emotion,
        confidence=confidence_score,
        music_mood=EMOTION_TO_MUSIC_MOOD[emotion],
        all_probabilities=all_probabilities,
        status="low_confidence" if confidence_score < 0.70 else "success",
        warning="Low confidence - emotion prediction may be uncertain" if confidence_score < 0.70 else None
    )
```

### Step 4: Update `/predict-realtime/` endpoint

**Before:**
```python
@app.post("/predict-realtime/")
async def predict_realtime(audio_buffer: UploadFile = File(...)):
    # ... processing ...
    
    return {
        "emotion": emotion,
        "confidence": round(confidence.item(), 4),
        "music_mood": EMOTION_TO_MUSIC_MOOD[emotion]
    }
```

**After:**
```python
@app.post("/predict-realtime/", response_model=RealtimePredictionResponse)
async def predict_realtime(audio_buffer: UploadFile = File(...)):
    # ... processing ...
    
    return RealtimePredictionResponse(
        emotion=emotion,
        confidence=confidence.item(),
        music_mood=EMOTION_TO_MUSIC_MOOD[emotion]
    )
```

## Benefits You Get Immediately

1. **FastAPI auto-generates OpenAPI docs** at `http://localhost:8000/docs`
   - Shows exact response schema
   - Includes field descriptions from Pydantic models
   - Provides "Try it out" functionality

2. **Validation happens automatically**
   ```python
   # This would raise ValidationError
   EmotionPredictionResponse(
       emotion="invalid_emotion",  # Must be one of: neutral, calm, happy...
       confidence=2.5,              # Must be between 0.0 and 1.0
       music_mood="rock",           # Must be one of: chill, relaxing...
       all_probabilities={}         # Required field
   )
   ```

3. **Type hints work in IDE**
   - Autocomplete for all fields
   - Catch typos before running
   - Know exactly what data structure to expect

4. **Guaranteed sync with mobile app**
   - Same structure defined in `api-contracts.ts`
   - If backend changes response, TypeScript app knows immediately

## Testing Example

```python
# Test with pytest
def test_emotion_prediction():
    response = client.post(
        "/predict-emotion/",
        files={"file": ("test.wav", audio_data, "audio/wav")}
    )
    
    # Pydantic ensures this structure
    data = EmotionPredictionResponse(**response.json())
    assert data.confidence >= 0.0 and data.confidence <= 1.0
    assert data.emotion in ["neutral", "calm", "happy", ...]
    assert data.music_mood is not None
```

## Summary

**Currently:** Manual JSON construction, no validation, easy to make mistakes  
**With Pydantic:** Type-safe, auto-validated, auto-documented, IDE-friendly

Your `models.py` is ready to use - you just need to import it and replace `JSONResponse(content={...})` with `EmotionPredictionResponse(...)`.
