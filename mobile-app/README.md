# Mobile App (Coming Soon)

React Native cross-platform app for iOS, Android, and Web.

## 🎯 Planned Features

- Voice recording and emotion analysis
- Real-time emotion detection display
- Spotify integration for music recommendations
- Emotion history tracking
- Music mood visualization
- Cross-platform (iOS, Android, Web) with single codebase

## 📱 Tech Stack

- **Framework**: React Native + React Native Web
- **Language**: TypeScript
- **State Management**: Zustand or React Context
- **UI Components**: React Native Paper
- **Navigation**: React Navigation v6
- **API Client**: Axios
- **Audio**: react-native-audio-recorder-player

## 🚀 Setup (When Available)

### Prerequisites
- Node.js 18+
- React Native CLI
- iOS: Xcode, CocoaPods
- Android: Android Studio, JDK 11+

### Installation
```bash
cd mobile-app
npm install

# iOS
cd ios && pod install && cd ..
npm run ios

# Android
npm run android

# Web
npm run web
```

## 🔗 ML Backend Integration

The mobile app will consume the ML backend API:

```typescript
import { EMOTION_LABELS, EMOTION_TO_MUSIC_MOOD } from '@shared/constants/emotions';
import { EmotionPredictionResponse } from '@shared/types/api-contracts';

// Call ML API
const response = await fetch('http://localhost:8000/predict-emotion/', {
  method: 'POST',
  body: formData,
});

const result: EmotionPredictionResponse = await response.json();
console.log(`Detected emotion: ${result.emotion}`);
console.log(`Music mood: ${result.music_mood}`);
```

## 📂 Planned Structure

```
mobile-app/
├── src/
│   ├── components/        # Reusable UI components
│   │   ├── ToneAnalyzer/
│   │   ├── MusicPlayer/
│   │   └── EmotionDisplay/
│   ├── screens/           # App screens
│   │   ├── HomeScreen/
│   │   ├── RecordScreen/
│   │   └── HistoryScreen/
│   ├── services/          # API clients
│   │   ├── mlApi.ts      # ML backend API
│   │   └── spotifyApi.ts # Spotify API
│   ├── store/             # State management
│   ├── types/             # TypeScript types
│   └── App.tsx            # Root component
├── ios/                   # iOS native files
├── android/               # Android native files
├── web/                   # Web-specific config
├── package.json
└── tsconfig.json
```

## 🎨 UI/UX Mockup

```
┌─────────────────────────┐
│   🎤 Emotion Detector   │
├─────────────────────────┤
│                         │
│    [  Record Button  ]  │
│                         │
│  Detected: 😊 Happy     │
│  Confidence: 92%        │
│  Mood: Upbeat           │
│                         │
│  [View Spotify Playlist]│
│                         │
└─────────────────────────┘
```

## 📦 Environment Variables

```env
# ML Backend API
ML_API_URL=http://localhost:8000

# Spotify API
SPOTIFY_CLIENT_ID=your_client_id
SPOTIFY_CLIENT_SECRET=your_client_secret
SPOTIFY_REDIRECT_URI=your_redirect_uri
```

## 🔧 Configuration

### TypeScript Path Aliases
```json
{
  "compilerOptions": {
    "paths": {
      "@shared/*": ["../shared/*"],
      "@components/*": ["src/components/*"],
      "@screens/*": ["src/screens/*"]
    }
  }
}
```

## 🤝 Contributing

Once the mobile app is initialized, follow these guidelines:
1. Use functional components with hooks
2. Follow TypeScript strict mode
3. Use shared types from `@shared/types/api-contracts`
4. Test on iOS, Android, and Web before submitting PR

## 📝 Status

**Current**: Planning phase - folder structure created  
**Next Steps**:
1. Initialize React Native project
2. Set up TypeScript configuration
3. Implement audio recording
4. Integrate with ML backend API
5. Add Spotify authentication
6. Build UI components

Check back soon for updates!
