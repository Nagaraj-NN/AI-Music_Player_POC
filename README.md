# Emotion Music Player

Welcome to the Emotion Music Player project — a 3-week proof-of-concept for real-time speech emotion recognition integrated with music playback.

## Project Overview

This project creates a system that detects emotions from speech audio using convolutional neural networks (CNNs) and plays music playlists from Spotify matched to the detected emotions.

The workflow spans:

- **Week 1:** Load HuggingFace speech emotion datasets, build/train CNN models (1D on raw audio, 2D on Mel spectrograms, hybrid fusion), and export the best model.  
- **Week 2:** Implement real-time microphone audio capture with PyAudio, run live emotion inference, and create a Streamlit UI for interaction.  
- **Week 3:** Integrate Spotify API for emotion-based playlist search, auto-play, and deploy the app using Docker.

## Folder Structure

```text
emotion-music-player/
├── data/                       # Sample datasets and downloaded audio
├── src/
│   ├── hf_dataset_loader.py    # HuggingFace dataset loading
│   ├── emotion_cnn.py          # CNN architectures (1D, 2D, hybrid)
│   ├── train.py                # Model training and validation
│   ├── audio_capture.py        # Real-time microphone capture code
│   ├── spotify_client.py       # Spotify API wrapper & playlist search
│   └── emotion_inference.py    # Load model and predict emotions
├── models/
│   └── emotion_cnn.onnx        # Exported ONNX model for deployment
├── static/                     # UI assets like CSS
│   └── style.css
├── app.py                      # Streamlit UI entrypoint
├── spotify_config.py           # Spotify credentials (keep secret)
└── requirements.txt            # Python package dependencies
```

## Setup Instructions

1. **Clone this repository.**

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   
   # Linux/Mac
   source venv/bin/activate
   
   # Windows
   .\venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Spotify API:**
   Add your Spotify API credentials in `spotify_config.py`.

## Usage

### Week 1: Training
Run the training script to train your CNN emotion recognition model. The best model will be exported as `models/emotion_cnn.onnx`.

```bash
python src/train.py
```

### Week 2: Real-time Inference
Launch the Streamlit app for real-time emotion detection and visualization:

```bash
streamlit run app.py
```

- **Week 3:** Full app supports Spotify playlist search and auto-play based on detected emotions.

## Contributing

Feel free to open issues or PRs for improvements or bug fixes.

## License

Specify your license here (e.g. MIT License).
