/**
 * Shared TypeScript type definitions for ML API
 * Keep this in sync with Python Pydantic models
 */

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

export type MusicMoodType = 
  | 'chill'
  | 'relaxing'
  | 'upbeat'
  | 'uplifting'
  | 'energetic'
  | 'calming'
  | 'neutral'
  | 'exciting';

// ===== EMOTION PROBABILITIES =====
export interface EmotionProbabilities {
  neutral: number;
  calm: number;
  happy: number;
  sad: number;
  angry: number;
  fearful: number;
  disgust: number;
  surprised: number;
}

// ===== API RESPONSE TYPES =====
export interface EmotionPredictionResponse {
  emotion: EmotionType;
  confidence: number; // 0-1 range
  music_mood: MusicMoodType;
  all_probabilities: EmotionProbabilities;
  status: 'success' | 'low_confidence';
  warning?: string;
}

export interface RealtimePredictionResponse {
  emotion: EmotionType;
  confidence: number;
  music_mood: MusicMoodType;
}

// ===== API METADATA =====
export interface APIHealthResponse {
  status: 'running' | 'error';
  service: string;
  model_loaded: boolean;
  device: 'cuda' | 'cpu';
}

export interface APIError {
  detail: string;
  status_code: number;
  timestamp?: string;
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

export interface SpotifyTrack {
  trackId: string;
  name: string;
  artist: string;
  previewUrl: string;
  albumArt: string;
}
