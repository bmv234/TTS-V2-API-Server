from typing import Dict, List, Optional, Tuple, Literal
import io
import os
import wave
import base64
from pathlib import Path
from enum import Enum
from datetime import datetime

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download

# OpenAI API compatibility types
class AudioModel(str, Enum):
    tts_1 = "tts-1"
    tts_1_hd = "tts-1-hd"

class AudioResponseFormat(str, Enum):
    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"

class AudioSpeechRequest(BaseModel):
    model: AudioModel = Field(default=AudioModel.tts_1)
    input: str = Field(..., description="The text to generate audio for")
    voice: str = Field(..., description="The voice to use")
    response_format: AudioResponseFormat = Field(default=AudioResponseFormat.mp3)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

class AudioSpeechResponse(BaseModel):
    created: int = Field(..., description="Unix timestamp of when the audio was created")
    duration: float = Field(..., description="Duration of the audio in seconds")

# Voice mapping
OPENAI_VOICE_MAP = {
    "alloy": "am_adam",      # Neutral male
    "echo": "af_nicole",     # Soft female
    "fable": "bf_emma",      # British female
    "onyx": "bm_george",     # Deep male
    "nova": "af_bella",      # Energetic female
    "shimmer": "af_sarah"    # Clear female
}

# Constants
SAMPLE_RATE = 24000
MODEL_REPO = "hexgrad/Kokoro-82M"
REQUIRED_FILES = [
    'models.py',
    'kokoro.py',
    'istftnet.py',
    'plbert.py',
    'config.json',
    'kokoro-v0_19.pth',
    'voices/af.pt',
    'voices/af_bella.pt',
    'voices/af_sarah.pt',
    'voices/am_adam.pt',
    'voices/am_michael.pt',
    'voices/bf_emma.pt',
    'voices/bf_isabella.pt',
    'voices/bm_george.pt',
    'voices/bm_lewis.pt',
    'voices/af_nicole.pt'
]

VOICE_DESCRIPTIONS = {
    'af': 'Default voice (50-50 mix of Bella & Sarah)',
    'af_bella': 'American Female - Bella',
    'af_sarah': 'American Female - Sarah',
    'am_adam': 'American Male - Adam',
    'am_michael': 'American Male - Michael',
    'bf_emma': 'British Female - Emma',
    'bf_isabella': 'British Female - Isabella',
    'bm_george': 'British Male - George',
    'bm_lewis': 'British Male - Lewis',
    'af_nicole': 'American Female - Nicole (ASMR voice)'
}

app = FastAPI(
    title="Kokoro TTS API",
    description="Text-to-Speech API using the Kokoro-82M model with OpenAI compatibility",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
voicepacks: Dict[str, torch.Tensor] = {}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def check_files() -> List[str]:
    """Check if all required files are present.
    
    Returns:
        List[str]: List of missing files
    """
    return [f for f in REQUIRED_FILES if not os.path.exists(f)]

def download_model_files() -> bool:
    """Download all necessary files from Hugging Face.
    
    Returns:
        bool: True if all files were downloaded successfully, False otherwise
    """
    print("Downloading model files from Hugging Face...")
    try:
        os.makedirs('voices', exist_ok=True)
        
        for file_path in REQUIRED_FILES:
            print(f"Downloading {file_path}...")
            try:
                downloaded_path = hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=file_path,
                    local_dir=".",
                    local_dir_use_symlinks=False
                )
                print(f"Successfully downloaded {file_path}")
            except Exception as e:
                print(f"Error downloading {file_path}: {str(e)}")
                return False
        return True
    except Exception as e:
        print(f"Error downloading model files: {str(e)}")
        return False

def create_wav_file(audio: np.ndarray) -> io.BytesIO:
    """Create a WAV file in memory from audio data.
    
    Args:
        audio (np.ndarray): Audio data as numpy array
        
    Returns:
        io.BytesIO: Buffer containing WAV file data
    """
    # Normalize audio to 16-bit PCM range
    audio = np.clip(audio * 32768, -32768, 32767).astype(np.int16)
    
    # Create WAV file in memory
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio.tobytes())
    
    buffer.seek(0)
    return buffer

def split_text_into_chunks(text: str, max_tokens: int = 510) -> List[str]:
    """Split text into chunks that won't exceed the token limit.
    
    Args:
        text (str): Text to split
        max_tokens (int): Maximum tokens per chunk
        
    Returns:
        List[str]: List of text chunks
    """
    # Import here to ensure files are downloaded
    from kokoro import phonemize, tokenize
    
    # Split text into sentences (basic implementation)
    sentences = text.replace('!', '.!').replace('?', '.?').split('.')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        # Check token count for this sentence
        sentence_tokens = len(tokenize(phonemize(sentence, 'a')))
        
        if current_tokens + sentence_tokens <= max_tokens:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_tokens = sentence_tokens
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def concatenate_audio(audio_chunks: List[np.ndarray]) -> np.ndarray:
    """Concatenate multiple audio chunks with a small silence between them.
    
    Args:
        audio_chunks (List[np.ndarray]): List of audio arrays to concatenate
        
    Returns:
        np.ndarray: Concatenated audio
    """
    # Add 0.1 seconds of silence between chunks
    silence_samples = int(0.1 * SAMPLE_RATE)
    silence = np.zeros(silence_samples)
    
    result = []
    for i, chunk in enumerate(audio_chunks):
        result.append(chunk)
        if i < len(audio_chunks) - 1:  # Don't add silence after the last chunk
            result.append(silence)
    
    return np.concatenate(result)

@app.on_event("startup")
async def startup_event():
    """Initialize the model and voice packs on server startup."""
    global model, voicepacks
    try:
        # Check for missing files
        missing_files = check_files()
        if missing_files:
            print(f"Missing files: {missing_files}")
            print("Attempting to download missing files...")
            success = download_model_files()
            if not success:
                raise Exception("Failed to download model files")
            
            # Check again after download
            missing_files = check_files()
            if missing_files:
                raise Exception(f"Still missing files after download: {missing_files}")

        # Import here after files are downloaded
        from models import build_model
        
        # Load model
        model = build_model('kokoro-v0_19.pth', device)
        print("Model loaded successfully")
        
        # Load all voice packs
        for voice_name in VOICE_DESCRIPTIONS.keys():
            try:
                voicepacks[voice_name] = torch.load(f'voices/{voice_name}.pt', weights_only=True).to(device)
                print(f'Loaded voice: {voice_name}')
            except Exception as e:
                print(f'Failed to load voice {voice_name}: {str(e)}')
                
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to load TTS model")

@app.post("/tts")
async def text_to_speech(text: str, voice: str = 'af') -> StreamingResponse:
    """Generate speech from text.
    
    Args:
        text (str): Text to convert to speech
        voice (str, optional): Voice ID to use. Defaults to 'af'.
        
    Returns:
        StreamingResponse: WAV audio file
        
    Raises:
        HTTPException: If model is not initialized or voice is not found
    """
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if voice not in voicepacks:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{voice}' not found. Available voices: {list(voicepacks.keys())}"
        )
    
    try:
        # Import here to ensure files are downloaded
        from kokoro import generate
        
        print(f"Generating speech for text: {text} with voice: {voice}")
        
        # Split text into chunks if needed
        chunks = split_text_into_chunks(text)
        audio_chunks = []
        
        # Generate audio for each chunk
        for chunk in chunks:
            print(f"Processing chunk: {chunk}")
            audio, phonemes = generate(model, chunk, voicepacks[voice], lang=voice[0])
            audio_chunks.append(audio)
            print(f"Generated audio shape: {audio.shape}, phonemes: {phonemes}")
        
        # Concatenate chunks if needed
        if len(audio_chunks) > 1:
            audio = concatenate_audio(audio_chunks)
        else:
            audio = audio_chunks[0]
        
        # Create WAV file
        buffer = create_wav_file(audio)
        print(f"WAV file size: {buffer.getbuffer().nbytes} bytes")
        
        return StreamingResponse(buffer, media_type="audio/wav")
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.get("/voices")
async def list_voices() -> Dict[str, dict]:
    """Get list of available voices and their descriptions.
    
    Returns:
        Dict[str, dict]: Dictionary containing voice information
    """
    return {
        "voices": {
            voice_id: {
                "description": desc,
                "language": "American English" if voice_id.startswith('a') else "British English",
                "gender": "Female" if voice_id.endswith(('f', 'bella', 'sarah', 'emma', 'isabella', 'nicole'))
                         else "Male" if voice_id.endswith(('adam', 'michael', 'george', 'lewis'))
                         else "Mixed"
            }
            for voice_id, desc in VOICE_DESCRIPTIONS.items()
        },
        "loaded_voices": list(voicepacks.keys())
    }

@app.get("/v1/models")
async def list_models():
    """OpenAI-compatible endpoint to list available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": "tts-1",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "kokoro",
                "permission": [],
                "root": "tts-1",
                "parent": None
            },
            {
                "id": "tts-1-hd",
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "kokoro",
                "permission": [],
                "root": "tts-1-hd",
                "parent": None
            }
        ]
    }

@app.post("/v1/audio/speech")
async def create_speech(request: AudioSpeechRequest):
    """OpenAI-compatible endpoint to generate speech from text."""
    if not model:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Map OpenAI voice to internal voice
    internal_voice = OPENAI_VOICE_MAP.get(request.voice)
    if not internal_voice:
        raise HTTPException(
            status_code=400,
            detail=f"Voice '{request.voice}' not found. Available voices: {list(OPENAI_VOICE_MAP.keys())}"
        )
    
    if internal_voice not in voicepacks:
        raise HTTPException(
            status_code=500,
            detail=f"Internal voice '{internal_voice}' not loaded"
        )
    
    try:
        from kokoro import generate
        
        # Split text into chunks if needed
        chunks = split_text_into_chunks(request.input)
        audio_chunks = []
        
        # Generate audio for each chunk
        for chunk in chunks:
            print(f"Processing chunk: {chunk}")
            audio, phonemes = generate(
                model, 
                chunk, 
                voicepacks[internal_voice], 
                lang=internal_voice[0],
                speed=request.speed
            )
            audio_chunks.append(audio)
        
        # Concatenate chunks if needed
        if len(audio_chunks) > 1:
            audio = concatenate_audio(audio_chunks)
        else:
            audio = audio_chunks[0]
        
        # Create WAV file
        buffer = create_wav_file(audio)
        
        # Calculate duration in seconds
        duration = len(audio) / SAMPLE_RATE
        
        # For now, we only support WAV format
        if request.response_format != AudioResponseFormat.wav:
            raise HTTPException(
                status_code=400,
                detail=f"Currently only WAV format is supported. Requested: {request.response_format}"
            )
        
        # Return streaming response with proper headers
        headers = {
            "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
        }
        
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers=headers
        )
    
    except Exception as e:
        print(f"Error generating speech: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Speech generation failed: {str(e)}")

@app.get("/")
async def root() -> Dict[str, dict]:
    """Get API usage information."""
    return {
        "message": "Kokoro TTS API is running",
        "usage": {
            "openai_compatible": {
                "models": "GET /v1/models",
                "speech": "POST /v1/audio/speech"
            },
            "native": {
                "tts": "POST /tts",
                "voices": "GET /voices"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
