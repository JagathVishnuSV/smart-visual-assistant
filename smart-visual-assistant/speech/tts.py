"""
Text-to-Speech Module - Optimized for accessibility
Full voice interaction support for visually impaired and normal users
"""
import pyttsx3
import logging
import threading
import queue
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global engine instance for reuse
_engine = None
_engine_lock = threading.Lock()
_speech_queue = queue.Queue()
_is_speaking = False
_speech_thread = None


def get_engine():
    """Get or create TTS engine instance"""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                try:
                    _engine = pyttsx3.init()
                    # Configure for better accessibility
                    _engine.setProperty('rate', 160)  # Optimized for clarity
                    _engine.setProperty('volume', 1.0)
                    
                    # Try to set a better voice if available
                    voices = _engine.getProperty('voices')
                    if voices:
                        # Prefer female voice for better clarity (index 1 usually)
                        if len(voices) > 1:
                            _engine.setProperty('voice', voices[1].id)
                    
                    logger.info("TTS engine initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize TTS engine: {e}")
                    raise
    return _engine


def speak(text, interrupt=False):
    """
    Synchronous speech - blocks until complete
    
    Args:
        text: Text to speak
        interrupt: If True, stop current speech and speak immediately
    """
    if not text or not text.strip():
        return
    
    global _is_speaking
    
    try:
        if interrupt:
            stop_speaking()
        
        _is_speaking = True
        engine = get_engine()
        engine.say(text)
        engine.runAndWait()
        _is_speaking = False
        
    except Exception as e:
        logger.error(f"TTS error: {e}")
        print(f"[AUDIO]: {text}")  # Fallback to text output
        _is_speaking = False


def speak_async(text, interrupt=False):
    """
    Asynchronous speech - non-blocking
    
    Args:
        text: Text to speak
        interrupt: If True, stop current speech and speak immediately
    """
    if not text or not text.strip():
        return
    
    def _speak():
        speak(text, interrupt=interrupt)
    
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
    return thread


def stop_speaking():
    """Stop current speech"""
    global _is_speaking
    try:
        engine = get_engine()
        engine.stop()
        _is_speaking = False
    except Exception as e:
        logger.error(f"Failed to stop speech: {e}")


def is_speaking():
    """Check if TTS is currently speaking"""
    return _is_speaking


def speak_with_callback(text, callback=None):
    """
    Speak text and call callback when done
    
    Args:
        text: Text to speak
        callback: Function to call when speech is complete
    """
    def _speak_and_callback():
        speak(text)
        if callback:
            callback()
    
    thread = threading.Thread(target=_speak_and_callback, daemon=True)
    thread.start()
    return thread


def set_voice_properties(rate=160, volume=1.0, voice_index=None):
    """
    Configure voice properties
    
    Args:
        rate: Speech rate (words per minute, default 160)
        volume: Volume level (0.0 to 1.0)
        voice_index: Index of voice to use (None to keep current)
    """
    try:
        engine = get_engine()
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        
        if voice_index is not None:
            voices = engine.getProperty('voices')
            if 0 <= voice_index < len(voices):
                engine.setProperty('voice', voices[voice_index].id)
                logger.info(f"Voice changed to: {voices[voice_index].name}")
    except Exception as e:
        logger.error(f"Failed to set voice properties: {e}")


def get_available_voices():
    """Get list of available TTS voices"""
    try:
        engine = get_engine()
        voices = engine.getProperty('voices')
        return [(i, v.name, v.languages) for i, v in enumerate(voices)]
    except Exception as e:
        logger.error(f"Failed to get voices: {e}")
        return []


def speak_menu(options, prompt="Please choose an option"):
    """
    Speak a menu of options for accessibility
    
    Args:
        options: List of option strings
        prompt: Introduction prompt
    """
    text = f"{prompt}. "
    for i, option in enumerate(options, 1):
        text += f"Option {i}: {option}. "
    
    speak(text)
    