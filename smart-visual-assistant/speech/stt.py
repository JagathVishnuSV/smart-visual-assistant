"""
Speech-to-Text Module - Optimized for low-compute systems
Full speech recognition for visually impaired and normal users
Uses Whisper tiny model for efficiency
"""
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile
import logging
import os
import threading
import queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Use tiny model for low-compute systems
_model = None
_is_recording = False
_recorded_audio = None


def get_model():
    """Lazy load Whisper model"""
    global _model
    if _model is None:
        try:
            logger.info("Loading Whisper model (tiny for low-compute)...")
            _model = whisper.load_model("tiny")  # Much faster than small/base
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            raise
    return _model


def record_audio(duration=5, samplerate=16000):
    """
    Record audio from microphone
    
    Args:
        duration: Recording duration in seconds
        samplerate: Sample rate (16000 is optimal for Whisper)
    
    Returns:
        Audio data as numpy array
    """
    global _is_recording
    try:
        print(f"🎤 Listening for {duration} seconds...")
        _is_recording = True
        audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        _is_recording = False
        print("✅ Recording complete")
        return np.squeeze(audio)
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        _is_recording = False
        raise


def record_until_silence(max_duration=10, silence_threshold=0.01, silence_duration=2.0, samplerate=16000):
    """
    Record audio until silence is detected
    
    Args:
        max_duration: Maximum recording duration
        silence_threshold: Volume threshold for silence detection
        silence_duration: Duration of silence to stop recording
        samplerate: Sample rate
    
    Returns:
        Audio data as numpy array
    """
    global _is_recording
    try:
        print("🎤 Listening... (speak now, will stop after silence)")
        _is_recording = True
        
        chunk_duration = 0.1  # 100ms chunks
        chunks = []
        silent_chunks = 0
        required_silent_chunks = int(silence_duration / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        
        for _ in range(max_chunks):
            chunk = sd.rec(int(chunk_duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
            sd.wait()
            chunk = np.squeeze(chunk)
            chunks.append(chunk)
            
            # Check if silent
            if np.abs(chunk).mean() < silence_threshold:
                silent_chunks += 1
                if silent_chunks >= required_silent_chunks:
                    print("✅ Silence detected, stopping recording")
                    break
            else:
                silent_chunks = 0
        
        _is_recording = False
        audio = np.concatenate(chunks)
        return audio
        
    except Exception as e:
        logger.error(f"Recording failed: {e}")
        _is_recording = False
        raise


def transcribe_audio(duration=5, language='en'):
    """
    Record and transcribe audio
    
    Args:
        duration: Recording duration in seconds
        language: Language code (e.g., 'en', 'es', 'fr')
    
    Returns:
        Transcribed text string
    """
    try:
        audio = record_audio(duration=duration)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            scipy.io.wavfile.write(temp_path, 16000, (audio * 32767).astype(np.int16))
        
        # Transcribe
        model = get_model()
        result = model.transcribe(temp_path, language=language, fp16=False)  # CPU mode
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass
        
        text = result["text"].strip()
        logger.info(f"Transcribed: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""


def transcribe_until_silence(max_duration=10, language='en'):
    """
    Record until silence and transcribe
    
    Args:
        max_duration: Maximum recording duration
        language: Language code
    
    Returns:
        Transcribed text string
    """
    try:
        audio = record_until_silence(max_duration=max_duration)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            scipy.io.wavfile.write(temp_path, 16000, (audio * 32767).astype(np.int16))
        
        # Transcribe
        model = get_model()
        result = model.transcribe(temp_path, language=language, fp16=False)
        
        # Cleanup
        try:
            os.unlink(temp_path)
        except:
            pass
        
        text = result["text"].strip()
        logger.info(f"Transcribed: {text}")
        return text
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return ""


def transcribe_file(file_path, language='en'):
    """
    Transcribe audio from file
    
    Args:
        file_path: Path to audio file
        language: Language code
    
    Returns:
        Transcribed text string
    """
    try:
        model = get_model()
        result = model.transcribe(file_path, language=language, fp16=False)
        return result["text"].strip()
    except Exception as e:
        logger.error(f"File transcription failed: {e}")
        return ""


def listen_for_command(wake_word=None, timeout=30):
    """
    Listen for voice command with optional wake word
    
    Args:
        wake_word: Optional wake word to activate (e.g., "assistant")
        timeout: Maximum time to wait for wake word
    
    Returns:
        Transcribed command text
    """
    try:
        if wake_word:
            print(f"💤 Waiting for wake word '{wake_word}'...")
            
            # Listen for wake word
            start_time = time.time()
            while time.time() - start_time < timeout:
                text = transcribe_audio(duration=3)
                if wake_word.lower() in text.lower():
                    print(f"✅ Wake word detected!")
                    from speech.tts import speak
                    speak("Yes, I'm listening")
                    break
            else:
                print("⏱️ Timeout waiting for wake word")
                return ""
        
        # Get actual command
        return transcribe_until_silence()
        
    except Exception as e:
        logger.error(f"Command listening failed: {e}")
        return ""


def is_recording_active():
    """Check if currently recording"""
    return _is_recording


def test_microphone():
    """Test microphone and audio recording"""
    try:
        print("🎤 Testing microphone...")
        audio = record_audio(duration=2)
        max_amplitude = np.abs(audio).max()
        
        if max_amplitude < 0.01:
            print("⚠️ Warning: Very low audio level. Check microphone volume.")
            return False
        elif max_amplitude > 0.9:
            print("⚠️ Warning: Audio might be clipping. Reduce microphone volume.")
            return True
        else:
            print(f"✅ Microphone working! Max amplitude: {max_amplitude:.3f}")
            return True
            
    except Exception as e:
        logger.error(f"Microphone test failed: {e}")
        print("❌ Microphone test failed. Check your audio settings.")
        return False