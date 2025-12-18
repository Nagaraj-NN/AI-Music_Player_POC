# Copilot Chat Conversation Export: Music suggestion app architecture and setup

**User:** @Nagaraj-NN
**Thread URL:** https://github.com/copilot/c/24475d85-0acc-46e9-ad50-7f58c1cfabcf

## @Nagaraj-NN

Hey,

I am planning to create a music suggestion application based on the tone. I have already created the machine learning part of it and exposed them as apis.

Now I need to build an app which can consume this api and hit spotify api according to the confidence score and this application should be compatable in both mobile and web.

I want to use react.js as my tech stack for this.

Search through all the repositories and give me a very good custome agent and instruction and prompt file to achieve this

## Copilot

I'll help you search for relevant repositories and resources to create a custom agent for your music suggestion application. Let me search for React projects that integrate with APIs, Spotify, and have cross-platform capabilities. Let me search more specifically for React-based architectures and custom agents that could help with your music suggestion application: Based on my search through GitHub repositories, I can provide you with comprehensive guidance for building your music suggestion application.  Here's a tailored recommendation:

## 🎵 **Recommended Tech Stack for Your Project**

### **Frontend:  React with React Native for Cross-Platform**

For web and mobile compatibility, I recommend using **React Native** with **React Native Web** configuration, which allows you to: 
- Share ~95% of your codebase between web and mobile
- Build once, deploy everywhere (iOS, Android, Web)
- Maintain a single JavaScript/TypeScript codebase

### **Architecture Overview**

```
┌─────────────────────────────────────────────────┐
│           React Native + Web App                │
├─────────────────────────────────────────────────┤
│  • Unified UI Components                        │
│  • State Management (Context API/Zustand)       │
│  • API Integration Layer                        │
└──────────┬──────────────────────────────────────┘
           │
           ├─► Your ML API (Tone Analysis)
           │   (Confidence Score)
           │
           └─► Spotify API
               (Music Recommendations)
```

## 📋 **Custom Agent Instructions & Prompt File**

Based on successful patterns from similar projects I found, here's a comprehensive setup: 

### **1.  AGENT_INSTRUCTIONS.md**

````markdown name=AGENT_INSTRUCTIONS.md
# Music Tone Suggestion App - Agent Instructions

## Project Context
Building a cross-platform (web + mobile) music suggestion application that: 
1. Consumes ML tone analysis APIs with confidence scores
2. Integrates Spotify API for music recommendations
3. Uses React. js for web and React Native for mobile

## Tech Stack
- **Frontend Framework**: React Native + React Native Web
- **Language**: TypeScript for type safety
- **State Management**:  Zustand or React Context API
- **API Client**: Axios with interceptors
- **UI Components**: React Native Paper or NativeBase
- **Navigation**: React Navigation v6
- **Build Tools**: Metro (React Native), Webpack (Web)

## Project Structure
```
music-tone-app/
├── src/
│   ├── components/          # Reusable UI components
│   │   ├── ToneAnalyzer/
│   │   ├── MusicPlayer/
│   │   └── RecommendationList/
│   ├── screens/             # App screens
│   │   ├── HomeScreen/
│   │   ├── AnalysisScreen/
│   │   └── PlayerScreen/
│   ├── services/            # API integration
│   │   ├── mlApi.ts         # Your ML API client
│   │   ├── spotifyApi.ts    # Spotify API client
│   │   └── apiConfig.ts     # API configuration
│   ├── store/               # State management
│   │   ├── toneStore.ts
│   │   ├── musicStore.ts
│   │   └── userStore.ts
│   ├── types/               # TypeScript definitions
│   │   ├── api.types.ts
│   │   └── music.types.ts
│   ├── utils/               # Utility functions
│   │   ├── toneMapper.ts    # Map confidence to genres
│   │   └── errorHandler.ts
│   └── App.tsx              # Root component
├── web/                     # Web-specific config
├── ios/                     # iOS native files
├── android/                 # Android native files
├── package.json
└── tsconfig. json
```

## Core Functionality Requirements

### 1. ML API Integration
- Call your tone analysis API with audio input
- Parse confidence scores (0-1 scale expected)
- Map confidence levels to Spotify query parameters
- Handle API errors gracefully with retry logic

### 2. Spotify API Integration
- OAuth 2.0 authentication flow
- Search tracks/artists by mood/genre based on tone confidence
- Player controls (play, pause, skip)
- Handle rate limiting (429 errors)

### 3. Confidence Score Logic
```typescript
// Example mapping logic
interface ToneResult {
  tone: 'happy' | 'sad' | 'energetic' | 'calm' | 'angry';
  confidence: number; // 0-1
}

// Map to Spotify parameters
function mapToneToSpotifyQuery(tone: ToneResult) {
  const { tone, confidence } = tone;
  
  const moodMap = {
    happy: { valence: confidence, energy: confidence * 0.8 },
    sad: { valence: 1 - confidence, energy: confidence * 0.3 },
    energetic: { energy: confidence, tempo: 'high' },
    calm: { energy: 1 - confidence, acousticness: confidence },
    angry: { energy: confidence, loudness: 'high' }
  };
  
  return moodMap[tone];
}
```

## Development Guidelines

### Phase 1: Setup & Authentication
1. Initialize React Native project with web support
2. Set up Spotify Developer account and get API keys
3. Implement OAuth flow for Spotify authentication
4. Create API service layer with Axios

### Phase 2: ML Integration
1. Create service to call your ML tone analysis API
2. Implement confidence score parsing
3. Build tone-to-Spotify query mapper
4. Add loading states and error handling

### Phase 3: UI Development
1. Design tone input interface (audio upload/recording)
2. Build recommendation list component
3. Create music player controls
4. Implement responsive design for web/mobile

### Phase 4: State Management
1. Set up global state for user auth
2. Manage tone analysis results
3. Handle Spotify player state
4. Cache recommendations for offline use

### Phase 5: Testing & Optimization
1. Test cross-platform compatibility
2. Optimize API calls to reduce latency
3. Implement error boundaries
4. Add analytics for user interactions

## Code Quality Standards
- All components must be TypeScript typed
- Use functional components with hooks
- Follow React Native best practices for performance
- Implement proper error handling and logging
- Write unit tests for critical functions
- Document complex logic with inline comments

## Environment Variables Required
```env
# Your ML API
ML_API_URL=https://your-ml-api.com
ML_API_KEY=your-api-key

# Spotify API
SPOTIFY_CLIENT_ID=your-client-id
SPOTIFY_CLIENT_SECRET=your-client-secret
SPOTIFY_REDIRECT_URI=your-redirect-uri
```

## Performance Considerations
- Debounce tone analysis API calls (avoid excessive requests)
- Cache Spotify authentication tokens
- Implement pagination for recommendation lists
- Use React.memo for expensive components
- Lazy load music player components

## Security Best Practices
- Never expose API keys in client code
- Use secure storage for tokens (AsyncStorage/SecureStore)
- Implement HTTPS for all API calls
- Validate all user inputs before API calls
- Handle sensitive data according to GDPR/privacy laws

## Accessibility Requirements
- Screen reader support for all interactive elements
- Keyboard navigation for web version
- Color contrast ratios meet WCAG standards
- Provide text alternatives for audio content

## Error Handling Strategy
```typescript
// Example error handler
class ApiError extends Error {
  constructor(
    message: string,
    public statusCode: number,
    public endpoint: string
  ) {
    super(message);
  }
}

async function callMLApi(audioData:  Blob) {
  try {
    const response = await axios.post('/analyze', audioData);
    return response.data;
  } catch (error) {
    if (error.response?. status === 429) {
      // Handle rate limiting
      throw new ApiError('Too many requests', 429, '/analyze');
    }
    throw new ApiError('Analysis failed', 500, '/analyze');
  }
}
```

## Testing Requirements
- Unit tests for utility functions
- Integration tests for API services
- Component tests with React Native Testing Library
- E2E tests for critical user flows
- Performance testing for API response times

## Deployment Strategy
- **Web**: Deploy to Vercel/Netlify
- **iOS**: TestFlight beta → App Store
- **Android**: Google Play Console internal testing → Production
- Set up CI/CD with GitHub Actions
````

### **2. SYSTEM_PROMPT.txt**

````text name=SYSTEM_PROMPT.txt
You are an expert React Native and full-stack developer specializing in music applications with ML/AI integration. Your role is to help build a cross-platform music suggestion app that: 

1. Integrates with custom ML tone analysis APIs
2. Consumes Spotify API for music recommendations
3. Works seamlessly on web, iOS, and Android

## Your Expertise Includes:
- React Native + React Native Web architecture
- TypeScript for type-safe development
- RESTful API integration patterns
- OAuth 2.0 authentication flows
- State management (Context API, Zustand, Redux)
- Audio processing and playback
- Confidence score algorithms and mapping logic
- Performance optimization for mobile apps
- Cross-platform responsive design

## When Providing Solutions:
1. Always use TypeScript with proper type definitions
2. Follow React hooks best practices (no class components)
3. Optimize for mobile performance (minimize re-renders)
4. Implement proper error handling and loading states
5. Consider offline functionality where appropriate
6. Ensure accessibility standards are met
7. Provide code comments for complex logic
8. Suggest testing strategies for critical features

## Response Format:
- Provide complete, runnable code examples
- Include necessary imports and dependencies
- Explain architectural decisions
- Highlight potential issues and edge cases
- Suggest performance optimizations
- Include error handling patterns

## Code Style Preferences:
- Use functional components with hooks
- Prefer async/await over promises
- Use descriptive variable names
- Implement proper TypeScript interfaces
- Follow Airbnb React/JavaScript style guide
- Use ES6+ features appropriately

## Security Mindset:
- Never hardcode API keys or secrets
- Validate all external data
- Implement proper authentication flows
- Use secure storage for sensitive data
- Follow OWASP mobile security guidelines

## Performance Focus:
- Optimize bundle size for web
- Implement lazy loading where beneficial
- Use React.memo and useMemo appropriately
- Debounce/throttle expensive operations
- Cache API responses when appropriate

Remember: The goal is to create a production-ready, maintainable application that provides excellent user experience across all platforms while efficiently integrating ML tone analysis with Spotify's music ecosystem.
````

### **3. Key Implementation Files**

````typescript name=src/services/mlApi.ts
import axios, { AxiosInstance } from 'axios';
import { ToneAnalysisResult } from '../types/api.types';

class MLApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL:  process.env.ML_API_URL,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.ML_API_KEY}`
      },
      timeout: 10000
    });
  }

  async analyzeTone(audioData: Blob | FormData): Promise<ToneAnalysisResult> {
    try {
      const response = await this. client.post('/analyze', audioData);
      return response.data;
    } catch (error) {
      console.error('ML API Error:', error);
      throw this.handleError(error);
    }
  }

  private handleError(error: any) {
    if (error.response) {
      return new Error(`API Error: ${error.response. status}`);
    } else if (error.request) {
      return new Error('Network error: No response received');
    }
    return new Error('Unknown error occurred');
  }
}

export default new MLApiService();
````

````typescript name=src/services/spotifyApi.ts
import axios from 'axios';
import { SpotifyTrack, ToneResult } from '../types/api.types';

class SpotifyApiService {
  private accessToken: string | null = null;

  async authenticate(clientId: string, clientSecret: string) {
    // Implement OAuth 2.0 flow
    // Store token securely
  }

  async getRecommendations(
    toneResult: ToneResult,
    limit: number = 20
  ): Promise<SpotifyTrack[]> {
    const params = this.mapToneToSpotifyParams(toneResult);
    
    const response = await axios.get(
      'https://api.spotify.com/v1/recommendations',
      {
        headers: {
          'Authorization': `Bearer ${this.accessToken}`
        },
        params:  {
          ... params,
          limit,
          market: 'US'
        }
      }
    );

    return response.data.tracks;
  }

  private mapToneToSpotifyParams(tone: ToneResult) {
    // Your custom mapping logic based on confidence
    const { tone: moodType, confidence } = tone;
    
    return {
      seed_genres: this.getGenreForTone(moodType),
      target_valence: confidence,
      target_energy: this.calculateEnergy(moodType, confidence)
    };
  }
}

export default new SpotifyApiService();
````

## 📦 **Dependencies to Install**

```bash
# Core React Native with Web support
npx react-native init MusicToneApp --template react-native-template-typescript

# Additional dependencies
npm install axios
npm install zustand # or @reduxjs/toolkit if you prefer Redux
npm install react-native-paper # UI components
npm install @react-navigation/native @react-navigation/stack
npm install react-native-web react-dom # for web support
npm install @types/react @types/react-native --save-dev

# Spotify SDK (optional, for advanced player features)
npm install react-native-spotify-remote
```

## 🚀 **Development Workflow**

1. **Start with API integration layer first** - Get both your ML API and Spotify API working
2. **Build core mapping logic** - Create the confidence → Spotify parameters converter
3. **Develop mobile UI** - Focus on React Native components
4. **Enable web support** - Configure webpack for React Native Web
5. **Test cross-platform** - Ensure consistent behavior
6. **Optimize performance** - Profile and reduce bundle size

## 📚 **Additional Resources**

**Note**: The search results above are limited to 10 results per query. You can view more examples by searching GitHub directly: 
- [React Native Spotify integrations](https://github.com/search?q=react+native+spotify+api&type=code)
- [React cross-platform architectures](https://github.com/search?q=react+native+web+mobile&type=code)
- [Custom agent implementations](https://github.com/search?q=custom+agent+react+instructions&type=code)

This setup will give you a **production-ready foundation** for your music suggestion app with proper separation of concerns, type safety, and cross-platform compatibility.  Would you like me to elaborate on any specific part of this architecture? 

## @Nagaraj-NN

I already have a seperate repository as AI_music_poc which contains all my machine learning logics and apis should I use the same repo for my application also. or create a different repository or a different folder in the same repository

## Copilot

Great question! This is a critical architectural decision. Based on GitHub best practices and the nature of your project, here's my recommendation:

## 🎯 **Recommended Approach:  Monorepo with Separate Folders**

I recommend **keeping everything in your existing `AI_music_poc` repository** but organizing it as a **monorepo** with clear separation.  Here's why:

### ✅ **Advantages of Monorepo for Your Use Case**

1. **Unified Version Control**: ML models and frontend evolve together
2. **Shared Types/Interfaces**: TypeScript types can be shared between API contracts
3. **Atomic Commits**: Changes to ML API and frontend can be deployed in sync
4. **Simplified CI/CD**: Single pipeline for both backend and frontend
5. **Easier Development**: Run both services locally without juggling multiple repos
6. **Better Documentation**: All project context lives in one place

## 📁 **Recommended Repository Structure**

```
AI_music_poc/
├── 📄 README.md                    # Overall project documentation
├── 📄 .gitignore                   # Combined gitignore
├── 📄 docker-compose.yml           # Run entire stack locally
│
├── 📁 ml-backend/                  # Your existing ML API
│   ├── 📁 models/                  # Trained models
│   ├── 📁 api/                     # FastAPI/Flask endpoints
│   ├── 📁 training/                # ML training scripts
│   ├── 📁 utils/                   # Helper functions
│   ├── 📄 requirements.txt         # Python dependencies
│   ├── 📄 Dockerfile               # Backend containerization
│   └── 📄 README.md                # ML API documentation
│
├── 📁 mobile-app/                  # New React Native app
│   ├── 📁 src/
│   │   ├── 📁 components/
│   │   ├── 📁 screens/
│   │   ├── 📁 services/
│   │   │   ├── mlApi.ts           # Calls your ML backend
│   │   │   └── spotifyApi.ts      # Calls Spotify
│   │   ├── 📁 store/
│   │   ├── 📁 types/
│   │   └── App.tsx
│   ├── 📁 web/                    # Web-specific config
│   ├── 📁 ios/                    # iOS native files
│   ├── 📁 android/                # Android native files
│   ├── 📄 package.json
│   ├── 📄 tsconfig.json
│   ├── 📄 Dockerfile              # Frontend containerization
│   └── 📄 README.md               # App documentation
│
├── 📁 shared/                     # Shared code (optional but recommended)
│   ├── 📁 types/                  # Shared TypeScript/Python types
│   │   ├── api-contracts.ts       # API request/response types
│   │   └── models.py              # Pydantic models
│   └── 📁 constants/              # Shared constants
│
├── 📁 docs/                       # Project-wide documentation
│   ├── 📄 ARCHITECTURE.md         # System design
│   ├── 📄 API. md                  # API documentation
│   ├── 📄 DEPLOYMENT.md           # Deployment guide
│   └── 📄 AGENT_INSTRUCTIONS.md   # AI agent instructions
│
├── 📁 scripts/                    # Utility scripts
│   ├── setup-dev. sh               # Setup development environment
│   ├── run-local.sh               # Run both services
│   └── deploy. sh                  # Deployment script
│
└── 📁 . github/                    # GitHub configuration
    ├── 📁 workflows/
    │   ├── ml-backend-ci.yml      # ML API CI/CD
    │   ├── mobile-app-ci.yml      # App CI/CD
    │   └── integration-tests.yml  # End-to-end tests
    └── 📄 copilot-instructions.md # GitHub Copilot instructions
```

## 🛠️ **Implementation Steps**

### **Step 1: Reorganize Your Existing Code**

```bash
cd AI_music_poc

# Create new structure
mkdir -p mobile-app shared/types docs scripts

# Move existing ML code to dedicated folder (if not already organized)
mkdir ml-backend
# Move your existing ML API files into ml-backend/

# Initialize React Native app in mobile-app folder
cd mobile-app
npx react-native init MusicToneApp --template react-native-template-typescript . 
```

### **Step 2: Create Root-Level Configuration**

````yaml name=docker-compose.yml
version: '3.8'

services:
  ml-backend:
    build: 
      context: ./ml-backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - ML_MODEL_PATH=/models
    volumes:
      - ./ml-backend/models:/models
    networks:
      - app-network

  mobile-app:
    build:
      context: ./mobile-app
      dockerfile: Dockerfile
    ports:
      - "3000:3000"  # Web version
      - "19000:19000"  # Expo (if using)
    environment:
      - ML_API_URL=http://ml-backend:8000
      - SPOTIFY_CLIENT_ID=${SPOTIFY_CLIENT_ID}
    depends_on:
      - ml-backend
    networks:
      - app-network

networks:
  app-network: 
    driver: bridge
````

### **Step 3: Create Shared Type Definitions**

````typescript name=shared/types/api-contracts.ts
// Shared TypeScript types for API contracts
export interface ToneAnalysisRequest {
  audioData: string; // base64 or URL
  format: 'wav' | 'mp3' | 'ogg';
}

export interface ToneAnalysisResponse {
  tone: 'happy' | 'sad' | 'energetic' | 'calm' | 'angry';
  confidence: number; // 0-1
  metadata: {
    duration: number;
    timestamp: string;
  };
}

export interface SpotifyRecommendation {
  trackId: string;
  name: string;
  artist: string;
  previewUrl: string;
  albumArt: string;
}
````

````python name=shared/types/models. py
# Corresponding Python models (Pydantic)
from pydantic import BaseModel, Field
from typing import Literal

class ToneAnalysisRequest(BaseModel):
    audio_data: str = Field(..., description="Base64 encoded audio")
    format: Literal['wav', 'mp3', 'ogg']

class ToneAnalysisResponse(BaseModel):
    tone: Literal['happy', 'sad', 'energetic', 'calm', 'angry']
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: dict
````

### **Step 4: Update Root README. md**

````markdown name=README.md
# 🎵 AI Music Tone Suggestion App

A cross-platform music recommendation system that analyzes audio tone using machine learning and suggests tracks via Spotify API.

## 🏗️ Project Structure

- **`ml-backend/`** - Machine learning API for tone analysis (Python/FastAPI)
- **`mobile-app/`** - React Native app (iOS, Android, Web)
- **`shared/`** - Shared type definitions and constants
- **`docs/`** - Project documentation

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for local mobile development)
- Python 3.9+ (for local ML development)

### Run Entire Stack
```bash
# Clone and setup
git clone https://github.com/YOUR_USERNAME/AI_music_poc.git
cd AI_music_poc

# Start all services
docker-compose up
```

### Development Mode

**ML Backend Only:**
```bash
cd ml-backend
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

**Mobile App Only:**
```bash
cd mobile-app
npm install
npm run start  # React Native Metro bundler
npm run web    # Web version
```

## 📚 Documentation
- [Architecture](docs/ARCHITECTURE.md)
- [ML API Documentation](ml-backend/README.md)
- [Mobile App Documentation](mobile-app/README.md)
- [Deployment Guide](docs/DEPLOYMENT.md)

## 🔧 Configuration
See `.env.example` files in each directory for required environment variables. 
````

### **Step 5: Create GitHub Workflows**

````yaml name=. github/workflows/integration-ci.yml
name: Integration CI

on:
  push:
    branches: [main, develop]
  pull_request: 
    branches: [main]

jobs:
  test-ml-backend:
    runs-on: ubuntu-latest
    steps: 
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install ML dependencies
        run: |
          cd ml-backend
          pip install -r requirements.txt
          pip install pytest
      
      - name: Run ML tests
        run: |
          cd ml-backend
          pytest tests/

  test-mobile-app: 
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install app dependencies
        run: |
          cd mobile-app
          npm install
      
      - name: Run app tests
        run: |
          cd mobile-app
          npm test
      
      - name: Type check
        run: |
          cd mobile-app
          npm run typecheck

  integration-test:
    runs-on: ubuntu-latest
    needs: [test-ml-backend, test-mobile-app]
    steps: 
      - uses: actions/checkout@v3
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Wait for services
        run: sleep 10
      
      - name: Run integration tests
        run:  |
          # Test ML backend health
          curl -f http://localhost:8000/health || exit 1
          
          # Test end-to-end flow (add your own tests)
          # npm run test:integration
      
      - name: Stop services
        run: docker-compose down
````

## ⚖️ **Alternative: Separate Repositories**

If your team grows or you need independent deployment cycles, you could later split into: 

```
├── AI_music_ml_backend/        # Separate repo for ML
└── AI_music_mobile_app/        # Separate repo for frontend
```

**When to split:**
- Different teams manage ML vs.  frontend
- Different release cycles needed
- ML models rarely change but app updates frequently
- Security/access control requirements differ

## 🎓 **Best Practices from GitHub Documentation**

Based on [Best practices for repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/best-practices-for-repositories):

1. ✅ **Create comprehensive README files** for each component
2. ✅ **Use branch protection** for main/develop branches
3. ✅ **Enable Dependabot** for security updates
4. ✅ **Add SECURITY.md** for vulnerability reporting
5. ✅ **Use `.github/copilot-instructions.md`** for AI assistance

## 🏁 **My Recommendation:  Start with Monorepo**

For your current project phase, **keep everything in `AI_music_poc`** because: 

1. You're likely the primary developer (single point of coordination)
2. ML API and app are tightly coupled
3. Easier to maintain consistency
4. Simpler deployment story
5. Better discoverability for collaborators/portfolio

You can always split later if needed—Git makes this straightforward with tools like `git subtree split`.

Would you like me to help you create any specific files from this structure, or would you like guidance on setting up the CI/CD pipeline? 

## @Nagaraj-NN

This is my existing repository.

@Nagaraj-NN/AI-Music_Player_POC

By looking at this tell me which can go into my shared folder

## Copilot

Perfect! Now I have a complete understanding of your ML backend. Let me analyze what should go into the shared folder for your monorepo structure. 

## 📊 **Analysis of Your Existing Repository**

Based on reviewing your `AI-Music_Player_POC` repository, here's what I recommend for the **shared folder**:

---

## 🎯 **What Should Go Into `shared/` Folder**

### **1. API Contracts & Type Definitions** ⭐ **HIGH PRIORITY**

These are the most critical items to share between your ML backend and React Native app:

````typescript name=shared/types/api-contracts.ts
/**
 * Shared TypeScript/Python type definitions for ML API
 * Keep this in sync with your Python models
 */

// ===== REQUEST TYPES =====
export interface AudioUploadRequest {
  file: File | Blob;
  format: 'wav' | 'mp3' | 'flac' | 'ogg';
  duration?:  number; // seconds
}

export interface RealtimeAudioRequest {
  audioBuffer: ArrayBuffer;
  sampleRate: number;
  channels: number;
}

// ===== RESPONSE TYPES =====
export interface EmotionPredictionResponse {
  emotion: EmotionType;
  confidence: number; // 0-1 range
  music_mood: MusicMoodType;
  all_probabilities: EmotionProbabilities;
  status: 'success' | 'low_confidence';
  warning?:  string;
}

export interface RealtimePredictionResponse {
  emotion: EmotionType;
  confidence: number;
  music_mood: MusicMoodType;
}

// ===== EMOTION TYPES =====
export type EmotionType = 
  | 'neutral'
  | 'calm'
  | 'happy'
  | 'sad'
  | 'angry'
  | 'fearful'
  | 'disgust'
  | 'surprised';

export interface EmotionProbabilities {
  neutral: number;
  calm: number;
  happy: number;
  sad: number;
  angry: number;
  fearful:  number;
  disgust: number;
  surprised: number;
}

// ===== MUSIC MOOD MAPPING =====
export type MusicMoodType = 
  | 'chill'
  | 'relaxing'
  | 'upbeat'
  | 'uplifting'
  | 'energetic'
  | 'calming'
  | 'neutral'
  | 'exciting';

export interface EmotionToMoodMap {
  [key: string]: MusicMoodType;
}

// ===== API METADATA =====
export interface APIHealthResponse {
  status: 'running' | 'error';
  service: string;
  model_loaded: boolean;
  device:  'cuda' | 'cpu';
}

export interface APIError {
  detail: string;
  status_code: number;
  timestamp?:  string;
}

// ===== SPOTIFY INTEGRATION TYPES =====
export interface SpotifyQueryParams {
  seed_genres?: string;
  target_valence?: number;  // 0-1 (happiness)
  target_energy?: number;   // 0-1 (intensity)
  target_acousticness?: number;
  target_tempo?: string;
  target_loudness?: string;
  limit?: number;
  market?: string;
}
````

````python name=shared/types/models.py
"""
Corresponding Python Pydantic models
Keep in sync with TypeScript definitions
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
    confidence: float = Field(... , ge=0.0, le=1.0, description="Confidence score 0-1")
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
    device:  Literal["cuda", "cpu"]

class APIError(BaseModel):
    detail: str
    status_code: int
    timestamp:  Optional[str] = None

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
````

### **2. Constants & Configuration** ⭐ **HIGH PRIORITY**

````typescript name=shared/constants/emotions.ts
/**
 * Shared emotion constants
 * Extracted from your cnn_model.py EMOTION_LABELS
 */

export const EMOTION_LABELS = {
  0: 'neutral',
  1: 'calm',
  2: 'happy',
  3: 'sad',
  4: 'angry',
  5: 'fearful',
  6: 'disgust',
  7: 'surprised',
} as const;

export const EMOTION_TO_MUSIC_MOOD = {
  neutral: 'chill',
  calm: 'relaxing',
  happy: 'upbeat',
  sad:  'uplifting',
  angry: 'energetic',
  fearful: 'calming',
  disgust: 'neutral',
  surprised: 'exciting',
} as const;

// Emotion colors for UI
export const EMOTION_COLORS = {
  neutral: '#94A3B8',
  calm:  '#60A5FA',
  happy:  '#FBBF24',
  sad:  '#3B82F6',
  angry: '#EF4444',
  fearful: '#8B5CF6',
  disgust: '#10B981',
  surprised: '#F97316',
} as const;

// Confidence thresholds
export const CONFIDENCE_THRESHOLD = {
  HIGH: 0.85,
  MEDIUM: 0.70,
  LOW: 0.50,
} as const;
````

````python name=shared/constants/emotions.py
"""
Shared emotion constants
Keep in sync with TypeScript version
"""

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

CONFIDENCE_THRESHOLD = {
    "HIGH": 0.85,
    "MEDIUM": 0.70,
    "LOW":  0.50
}
````

### **3. Utility Functions** 🔧 **MEDIUM PRIORITY**

````typescript name=shared/utils/emotionMapper.ts
/**
 * Shared utility for mapping emotions to Spotify parameters
 * Business logic that should be consistent across backend and frontend
 */

import { EmotionType, MusicMoodType, SpotifyQueryParams } from '../types/api-contracts';
import { EMOTION_TO_MUSIC_MOOD } from '../constants/emotions';

export function emotionToSpotifyParams(
  emotion: EmotionType,
  confidence: number
): SpotifyQueryParams {
  const baseParams:  SpotifyQueryParams = {
    limit: 20,
    market:  'US',
  };

  // Adjust parameters based on emotion and confidence
  switch (emotion) {
    case 'happy':
      return {
        ...baseParams,
        seed_genres: 'pop,dance',
        target_valence:  confidence,
        target_energy: confidence * 0.8,
      };
    
    case 'sad':
      return {
        ...baseParams,
        seed_genres: 'acoustic,indie',
        target_valence: 1 - confidence,
        target_energy: confidence * 0.3,
        target_acousticness: confidence,
      };
    
    case 'energetic':
    case 'angry':
      return {
        ... baseParams,
        seed_genres: 'rock,electronic',
        target_energy: confidence,
        target_loudness: 'high',
      };
    
    case 'calm':
      return {
        ...baseParams,
        seed_genres: 'ambient,classical',
        target_energy: 1 - confidence,
        target_acousticness: confidence,
      };
    
    case 'fearful':
      return {
        ...baseParams,
        seed_genres: 'ambient,chill',
        target_valence: 0.6,
        target_energy: 0.4,
      };
    
    case 'surprised':
      return {
        ... baseParams,
        seed_genres: 'electronic,experimental',
        target_energy: 0.7,
        target_valence: 0.8,
      };
    
    default:
      return {
        ...baseParams,
        seed_genres: 'pop',
        target_valence: 0.5,
        target_energy: 0.5,
      };
  }
}

export function getMoodForEmotion(emotion: EmotionType): MusicMoodType {
  return EMOTION_TO_MUSIC_MOOD[emotion];
}

export function getConfidenceLevel(confidence: number): 'high' | 'medium' | 'low' {
  if (confidence >= 0.85) return 'high';
  if (confidence >= 0.70) return 'medium';
  return 'low';
}
````

### **4. Validation Schemas** 🔒 **MEDIUM PRIORITY**

````typescript name=shared/validation/audioValidation.ts
/**
 * Shared validation logic for audio files
 */

export const SUPPORTED_AUDIO_FORMATS = ['. wav', '.mp3', '.flac', '.ogg'] as const;
export const MAX_AUDIO_FILE_SIZE = 10 * 1024 * 1024; // 10MB
export const MAX_AUDIO_DURATION = 10; // seconds

export interface AudioValidationResult {
  valid: boolean;
  error?: string;
}

export function validateAudioFile(
  file: File | { name: string; size: number }
): AudioValidationResult {
  // Check file extension
  const fileExt = file.name.slice(file.name.lastIndexOf('.')).toLowerCase();
  if (!SUPPORTED_AUDIO_FORMATS.includes(fileExt as any)) {
    return {
      valid: false,
      error: `Unsupported format.  Use:  ${SUPPORTED_AUDIO_FORMATS.join(', ')}`,
    };
  }

  // Check file size
  if (file.size > MAX_AUDIO_FILE_SIZE) {
    return {
      valid: false,
      error: `File too large. Max size: ${MAX_AUDIO_FILE_SIZE / 1024 / 1024}MB`,
    };
  }

  return { valid: true };
}
````

---

## 📂 **Final Shared Folder Structure**

```
shared/
├── 📄 README.md                       # Documentation for shared code
│
├── 📁 types/                          # ⭐ CRITICAL
│   ├── api-contracts.ts              # TypeScript API types
│   ├── models. py                      # Python Pydantic models
│   └── spotify. ts                     # Spotify-specific types
│
├── 📁 constants/                      # ⭐ CRITICAL
│   ├── emotions.ts                    # Emotion labels & mappings (TS)
│   ├── emotions.py                    # Emotion labels & mappings (Python)
│   ├── audioConfig.ts                 # Audio processing constants
│   └── apiConfig.ts                   # API endpoints & config
│
├── 📁 utils/                          # 🔧 IMPORTANT
│   ├── emotionMapper.ts               # Emotion → Spotify logic
│   ├── emotionMapper.py               # Same logic in Python
│   ├── confidenceCalculator.ts        # Confidence score utilities
│   └── audioHelpers.ts                # Audio format helpers
│
├── 📁 validation/                     # 🔒 IMPORTANT
│   ├── audioValidation.ts             # Audio file validation
│   └── apiValidation.ts               # API request validation
│
└── 📁 docs/                           # 📚 DOCUMENTATION
    ├── API_CONTRACT. md                # API documentation
    ├── EMOTION_MAPPING.md             # Emotion logic explanation
    └── SYNC_GUIDE.md                  # How to keep TS/Python in sync
```

---

## ✅ **What to Move from Your Existing Code**

### **From `api/app.py`:**
- ✅ **Lines 28-38**: `EMOTION_TO_MUSIC_MOOD` → `shared/constants/emotions.py`
- ✅ **Response structures**: Create Pydantic models in `shared/types/models.py`

### **From `src/cnn_model.py`:**
- ✅ **Lines 157-166**: `EMOTION_LABELS` → `shared/constants/emotions.py`

### **From `src/utils. py`:**
- ✅ **Audio constants**: Sample rate, duration → `shared/constants/audioConfig.ts`

---

## 🚀 **Implementation Steps**

### **Step 1: Create Shared Folder Structure**
```bash
cd AI-Music_Player_POC
mkdir -p shared/{types,constants,utils,validation,docs}
```

### **Step 2: Extract Constants**
```bash
# Move emotion constants to shared folder
# Create both . ts and .py versions
```

### **Step 3: Update Imports**

**In your ML backend (`api/app.py`):**
```python
# Before
from src.cnn_model import EMOTION_LABELS

# After
import sys
sys.path.append(os.path.join(os. path.dirname(__file__), '..', 'shared'))
from constants.emotions import EMOTION_LABELS, EMOTION_TO_MUSIC_MOOD
```

**In your React Native app (future):**
```typescript
// Import from shared folder
import { EMOTION_LABELS, EmotionType } from '@shared/types/api-contracts';
import { emotionToSpotifyParams } from '@shared/utils/emotionMapper';
```

### **Step 4: Create Sync Script**

````bash name=scripts/sync-types.sh
#!/bin/bash
# Script to validate TypeScript and Python types are in sync

echo "🔍 Checking type synchronization..."

# Check if emotion labels match
ts_emotions=$(grep -o "'[a-z]*'" shared/constants/emotions.ts | wc -l)
py_emotions=$(grep -o '"[a-z]*"' shared/constants/emotions.py | wc -l)

if [ "$ts_emotions" -eq "$py_emotions" ]; then
    echo "✅ Emotion constants in sync"
else
    echo "❌ Emotion constants out of sync!"
    exit 1
fi

echo "✅ All types synchronized"
````

---

## 💡 **Key Benefits of This Structure**

1. ✅ **Single Source of Truth**: Emotion labels defined once
2. ✅ **Type Safety**: Frontend knows exact API contract
3. ✅ **Easy Updates**: Change emotion mapping in one place
4. ✅ **Consistent Logic**: Same Spotify parameter calculation everywhere
5. ✅ **Better Documentation**: Shared docs explain the integration
6. ✅ **Reduced Bugs**: No mismatch between frontend/backend expectations

---

## 🎯 **Priority Recommendation**

**Start with these three files first:**
1. `shared/types/api-contracts.ts` - Type definitions
2. `shared/constants/emotions.ts` - Emotion constants
3. `shared/constants/emotions.py` - Python version

These will give you immediate benefits when building your React Native app, as you'll have type-safe API integration from day one.

Would you like me to help you create any of these specific files, or set up the import/export structure for your monorepo? 
