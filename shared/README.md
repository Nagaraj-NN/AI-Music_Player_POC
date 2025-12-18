# Shared Libraries

This folder contains code and constants shared between the ML backend and mobile app to ensure consistency.

## Structure

```
shared/
├── constants/          # Shared constants
│   ├── emotions.py    # Python version (ML backend)
│   └── emotions.ts    # TypeScript version (mobile app)
├── types/             # Type definitions
│   ├── models.py      # Python Pydantic models
│   └── api-contracts.ts  # TypeScript interfaces
└── utils/             # Shared utilities (future)
```

## Key Files

### `constants/emotions.py` & `emotions.ts`
- **EMOTION_LABELS**: Maps emotion indices (0-7) to emotion names
- **EMOTION_TO_MUSIC_MOOD**: Maps emotions to music moods for Spotify
- **CONFIDENCE_THRESHOLD**: Defines high/medium/low confidence levels
- **EMOTION_COLORS**: UI color codes for each emotion

**Important**: These files MUST be kept in sync manually. Any change to emotion mappings must be reflected in both files.

### `types/models.py` & `api-contracts.ts`
- API request/response type definitions
- Ensures type safety between Python backend and TypeScript frontend
- Python uses Pydantic for validation, TypeScript for compile-time safety

## Usage

### Python (ML Backend)
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from constants.emotions import EMOTION_LABELS, EMOTION_TO_MUSIC_MOOD
from types.models import EmotionPredictionResponse
```

### TypeScript (Mobile App)
```typescript
import { EMOTION_LABELS, EMOTION_TO_MUSIC_MOOD } from '@shared/constants/emotions';
import { EmotionPredictionResponse } from '@shared/types/api-contracts';
```

## Synchronization

When updating emotion constants or types:
1. Update the Python version first
2. Update the TypeScript version to match
3. Test both ML backend and mobile app
4. Commit both files together

## Validation Script (Future)
A sync validation script will be added to ensure TypeScript and Python definitions remain consistent.
