"""
NLP Prompt Engine - AI-powered contextual responses
Uses environment variables for API keys (secure)
"""
import os
from typing import List
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to load from .env file
try:
    
    load_dotenv()
except ImportError:
    logger.warning("python-dotenv not installed. Using environment variables only.")

# Initialize Gemini API
_model = None


def get_model():
    """Get or create Gemini model instance"""
    global _model
    if _model is None:
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key or api_key == "your_api_key_here":
                logger.warning("GEMINI_API_KEY not set. Using fallback responses.")
                return None
            
            genai.configure(api_key=api_key)
            _model = genai.GenerativeModel("gemini-2.0-flash-exp")  # Latest fast model
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            return None
    
    return _model


def build_prompt(query: str, objects: List[str]) -> str:
    """Build prompt for LLM"""
    object_str = ", ".join(objects) if objects else "nothing specific"
    return f"""You are an AI assistant helping a visually impaired person understand their environment.

User's question: "{query}"
Objects detected by camera: {object_str}

Provide a helpful, clear, and concise response (2-3 sentences max) that:
1. Answers their question directly
2. Mentions relevant detected objects
3. Uses simple, accessible language

Response:"""


def get_llm_response(prompt: str) -> str:
    """Get response from Gemini API"""
    model = get_model()
    
    if model is None:
        # Fallback response when API is not available
        return "I can see the detected objects in your environment. Please set up the Gemini API key for detailed AI responses."
    
    try:
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "max_output_tokens": 150,  # Keep responses concise
            }
        )
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return "I'm having trouble connecting to the AI service. Please check your internet connection."


def get_contextual_response(query: str, objects: List[str]) -> str:
    """Get contextual AI response based on query and detected objects"""
    if not query or not query.strip():
        if objects:
            return f"I can see {len(objects)} type(s) of objects: {', '.join(objects)}"
        return "I don't see any specific objects in the image."
    
    # Build and send prompt
    prompt = build_prompt(query, objects)
    return get_llm_response(prompt)


def get_fallback_response(objects: List[str]) -> str:
    """Fallback response when API is unavailable"""
    if not objects:
        return "I don't detect any specific objects in the current view."
    
    count = len(objects)
    if count == 1:
        return f"I can see a {objects[0]} in the image."
    elif count == 2:
        return f"I can see a {objects[0]} and a {objects[1]}."
    else:
        return f"I can see {count} types of objects: {', '.join(objects[:3])}{'and more' if count > 3 else ''}."


# Legacy function name for compatibility
def dummy_llm_response(prompt: str) -> str:
    """Legacy function - use get_llm_response instead"""
    return get_llm_response(prompt)