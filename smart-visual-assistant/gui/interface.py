"""
Accessibility Vision Assistant - Gradio Interface

This module provides a user-friendly Gradio web interface for the Smart Visual Assistant,
designed to help visually impaired users understand their environment through:
- Object detection
- Face detection
- QR/Barcode scanning
- Text reading (OCR)
- Color identification
- AI-powered scene understanding
"""

import logging
import cv2
import numpy as np
import threading
from typing import Optional, Tuple, List

import gradio as gr

from vision.fastvlm import FastVLM
from nlp.prompt_engine import get_contextual_response

# Optional imports for speech features
try:
    from speech.tts import speak_async
except ImportError:
    speak_async = None

try:
    from speech.stt import transcribe_audio, transcribe_until_silence
except ImportError:
    transcribe_audio = None
    transcribe_until_silence = None

# Optional live stream imports
try:
    from vision.live_stream import LiveVideoStream, ContinuousSceneDescriber
except ImportError:
    LiveVideoStream = None
    ContinuousSceneDescriber = None

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Global instances
# ---------------------------------------------------------------------------
_fast_vlm: Optional[FastVLM] = None
live_stream = None
scene_describer = None
stream_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_model() -> FastVLM:
    """Get or create singleton FastVLM instance."""
    global _fast_vlm
    
    if _fast_vlm is None:
        logger.info("Loading FastVLM models...")
        _fast_vlm = FastVLM()
    
    return _fast_vlm


def _maybe_speak(message: str, enable_audio: bool) -> None:
    """Speak message if audio is enabled and available."""
    if not enable_audio:
        return
    
    if speak_async is None:
        logger.debug("speak_async unavailable; skipping audio")
        return
    
    try:
        speak_async(message)
    except Exception as exc:
        logger.warning("Audio playback failed: %s", exc)


# ---------------------------------------------------------------------------
# Processing pipelines
# ---------------------------------------------------------------------------

def process_image(
    image: np.ndarray,
    mode: str,
    question: str,
    enable_audio: bool,
) -> Tuple[np.ndarray, str]:
    """
    Process image based on selected analysis mode.
    
    Args:
        image: Input image as numpy array
        mode: Analysis mode to use
        question: Optional question for AI mode
        enable_audio: Whether to provide audio feedback
        
    Returns:
        Tuple of (annotated_image, description_text)
    """
    if image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8), "Please provide an image"
    
    try:
        model = get_model()
        
        if mode == "🎯 Object Detection":
            annotated, labels = model.detect_objects(image)
            description = (
                f"I can see: {', '.join(labels)}" if labels else "No objects detected"
            )
        
        elif mode == "👤 Face Detection":
            annotated, count = model.detect_faces(image)
            if count == 0:
                description = "No faces detected"
            elif count == 1:
                description = "I detected 1 face in the image"
            else:
                description = f"I detected {count} faces in the image"
        
        elif mode == "📱 QR/Barcode Scanner":
            annotated, codes = model.detect_barcodes(image)
            description = (
                f"Scanned codes: {', '.join(codes)}"
                if codes
                else "No barcodes or QR codes found"
            )
        
        elif mode == "📄 Text Reading (OCR)":
            annotated, text = model.extract_text(image)
            description = f"Detected text: {text}" if text else "No text found"
        
        elif mode == "🎨 Color Identification":
            annotated, color = model.detect_dominant_color(image)
            description = f"Dominant color: {color}"
        
        elif mode == "🤖 AI Scene Understanding":
            annotated, answer = model.answer_question(
                image, question or "What do you see?"
            )
            description = answer
        
        elif mode == "🔍 Complete Scene Analysis":
            # Comprehensive analysis
            annotated, labels = model.detect_objects(image)
            annotated, face_count = model.detect_faces(annotated)
            annotated, color = model.detect_dominant_color(annotated)
            
            parts: List[str] = []
            if labels:
                parts.append(f"Objects: {', '.join(labels)}")
            parts.append(f"Faces detected: {face_count}")
            parts.append(f"Main color: {color}")
            description = " | ".join(parts)
        
        else:
            logger.warning("Unknown mode: %s", mode)
            return image, "Unknown analysis mode"
        
        # Provide audio feedback if enabled
        _maybe_speak(description, enable_audio)
        
        return annotated, description
    
    except Exception as exc:
        logger.exception("Processing failed: %s", exc)
        error_msg = f"Error: {exc}"
        _maybe_speak("An error occurred during processing", enable_audio)
        return image, error_msg


def process_with_ai(
    image: np.ndarray,
    user_query: str,
    enable_audio: bool,
) -> Tuple[np.ndarray, str]:
    """
    AI-powered contextual analysis with natural language queries.
    
    Args:
        image: Input image
        user_query: User's natural language question
        enable_audio: Whether to provide audio feedback
        
    Returns:
        Tuple of (annotated_image, ai_response)
    """
    if image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8), "Please provide an image"
    
    try:
        model = get_model()
        
        # Detect objects for context
        annotated, detected_labels = model.detect_objects(image)
        
        # Get AI response based on query and detected objects
        response = get_contextual_response(user_query, detected_labels)
        
        # Provide audio feedback
        _maybe_speak(response, enable_audio)
        
        return annotated, response
    
    except Exception as exc:
        logger.exception("AI contextual analysis failed: %s", exc)
        error_msg = f"Error: {exc}"
        return image, error_msg


# ---------------------------------------------------------------------------
# Live streaming functions (optional)
# ---------------------------------------------------------------------------

def start_live_stream(description_interval: int = 5) -> str:
    """Start live video stream with continuous scene description."""
    global live_stream, scene_describer
    
    if LiveVideoStream is None or ContinuousSceneDescriber is None:
        return "❌ Live stream feature not available (missing dependencies)"
    
    try:
        with stream_lock:
            if live_stream and live_stream.is_running:
                return "Live stream already running"
            
            model = get_model()
            
            # Create scene describer
            scene_describer = ContinuousSceneDescriber(
                vlm_model=model,
                speak_func=speak_async,
                min_description_interval=description_interval
            )
            
            # Create and start live stream
            live_stream = LiveVideoStream(camera_index=0, fps=2, width=640, height=480)
            
            if live_stream.start(analysis_callback=scene_describer.analyze_and_describe):
                if speak_async:
                    speak_async("Live stream started. I will describe what I see.")
                return "✅ Live stream started successfully"
            else:
                return "❌ Failed to start live stream. Check camera connection."
    
    except Exception as exc:
        logger.error("Failed to start live stream: %s", exc)
        return f"❌ Error: {str(exc)}"


def stop_live_stream() -> str:
    """Stop live video stream."""
    global live_stream, scene_describer
    
    try:
        with stream_lock:
            if live_stream:
                live_stream.stop()
                live_stream = None
                scene_describer = None
                if speak_async:
                    speak_async("Live stream stopped")
                return "✅ Live stream stopped"
            else:
                return "No active stream to stop"
    
    except Exception as exc:
        logger.error("Failed to stop live stream: %s", exc)
        return f"❌ Error: {str(exc)}"


def get_live_frame() -> np.ndarray:
    """Get current frame from live stream."""
    global live_stream
    
    if live_stream and live_stream.is_running:
        frame = live_stream.get_current_frame()
        if frame is not None:
            return frame
    
    # Return blank frame if no stream
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "No Active Stream", (180, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return blank


# ---------------------------------------------------------------------------
# Voice interaction functions (optional)
# ---------------------------------------------------------------------------

def voice_question() -> str:
    """Get question via voice input."""
    if transcribe_until_silence is None:
        return "Voice input not available"
    
    try:
        if speak_async:
            speak_async("Please ask your question now")
        
        question = transcribe_until_silence(max_duration=10)
        
        if question:
            return question
        else:
            if speak_async:
                speak_async("I didn't hear anything. Please try again.")
            return ""
    
    except Exception as exc:
        logger.error("Voice question failed: %s", exc)
        return f"Error: {str(exc)}"


def voice_mode_select() -> str:
    """Select analysis mode via voice."""
    if transcribe_audio is None:
        return "🎯 Object Detection"
    
    try:
        modes = [
            "Object Detection",
            "Face Detection",
            "QR Barcode Scanner",
            "Text Reading",
            "Color Identification",
            "AI Scene Understanding",
            "Complete Scene Analysis"
        ]
        
        modes_emoji = ["🎯", "👤", "📱", "📄", "🎨", "🤖", "🔍"]
        
        prompt = "Please say the number of your choice. "
        for i, mode in enumerate(modes, 1):
            prompt += f"{i} for {mode}. "
        
        if speak_async:
            speak_async(prompt)
        
        response = transcribe_audio(duration=3)
        
        # Extract number from response
        numbers = ["one", "two", "three", "four", "five", "six", "seven"]
        for i in range(1, 8):
            if str(i) in response or numbers[i-1] in response.lower():
                selected_mode = f"{modes_emoji[i-1]} {modes[i-1]}"
                if speak_async:
                    speak_async(f"You selected {modes[i-1]}")
                return selected_mode
        
        if speak_async:
            speak_async("I didn't understand. Using object detection.")
        return "🎯 Object Detection"
    
    except Exception as exc:
        logger.error("Voice mode selection failed: %s", exc)
        return "🎯 Object Detection"


# ---------------------------------------------------------------------------
# UI Creation
# ---------------------------------------------------------------------------

def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Accessibility Vision Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🌟 Accessibility Vision Assistant
        ### Empowering visually impaired users to understand their environment
        
        **Features:**
        - 🎯 Real-time object detection
        - 👤 Face detection and counting
        - 📱 QR code and barcode scanning
        - 📄 Text reading (OCR)
        - 🎨 Color identification
        - 🤖 AI-powered scene understanding
        - 🔊 Audio feedback support
        """)
        
        # Quick Analysis Tab
        with gr.Tab("📸 Quick Analysis"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="Upload Image or Use Webcam",
                        sources=["upload", "webcam"],
                        type="numpy"
                    )
                    mode_selector = gr.Radio(
                        choices=[
                            "🎯 Object Detection",
                            "👤 Face Detection",
                            "📱 QR/Barcode Scanner",
                            "📄 Text Reading (OCR)",
                            "🎨 Color Identification",
                            "🤖 AI Scene Understanding",
                            "🔍 Complete Scene Analysis"
                        ],
                        label="Select Analysis Mode",
                        value="🎯 Object Detection"
                    )
                    question_input = gr.Textbox(
                        label="Optional: Ask a question about the image",
                        placeholder="e.g., What objects are on the table?",
                        value=""
                    )
                    audio_toggle = gr.Checkbox(label="🔊 Enable Audio Feedback", value=True)
                    
                    with gr.Row():
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary", size="lg")
                        if transcribe_audio is not None:
                            voice_mode_btn = gr.Button("🎤 Select Mode by Voice", size="sm")
                            voice_question_btn = gr.Button("🎤 Ask by Voice", size="sm")
                
                with gr.Column():
                    output_image = gr.Image(label="Annotated Result")
                    output_text = gr.Textbox(label="Description", lines=5)
            
            # Connect event handlers
            analyze_btn.click(
                fn=process_image,
                inputs=[input_image, mode_selector, question_input, audio_toggle],
                outputs=[output_image, output_text]
            )
            
            if transcribe_audio is not None:
                voice_mode_btn.click(
                    fn=voice_mode_select,
                    outputs=[mode_selector]
                )
                
                voice_question_btn.click(
                    fn=voice_question,
                    outputs=[question_input]
                )
        
        # AI Assistant Tab
        with gr.Tab("💬 AI Assistant"):
            gr.Markdown("### Ask questions about your surroundings")
            with gr.Row():
                with gr.Column():
                    ai_image = gr.Image(
                        label="Upload Image",
                        sources=["upload", "webcam"],
                        type="numpy"
                    )
                    ai_query = gr.Textbox(
                        label="What would you like to know?",
                        placeholder="e.g., What's in front of me? Is there any text? What colors do you see?",
                        lines=3
                    )
                    ai_audio = gr.Checkbox(label="🔊 Enable Audio Response", value=True)
                    
                    with gr.Row():
                        ai_btn = gr.Button("🤖 Ask AI", variant="primary", size="lg")
                        if transcribe_audio is not None:
                            ai_voice_btn = gr.Button("🎤 Ask by Voice", size="sm")
                
                with gr.Column():
                    ai_output_image = gr.Image(label="Analyzed Image")
                    ai_output_text = gr.Textbox(label="AI Response", lines=8)
            
            ai_btn.click(
                fn=process_with_ai,
                inputs=[ai_image, ai_query, ai_audio],
                outputs=[ai_output_image, ai_output_text]
            )
            
            if transcribe_audio is not None:
                ai_voice_btn.click(
                    fn=voice_question,
                    outputs=[ai_query]
                )
        
        # Live Stream Tab (if available)
        if LiveVideoStream is not None:
            with gr.Tab("🎥 Live Video Stream"):
                gr.Markdown("### Continuous Real-Time Environment Analysis")
                
                with gr.Row():
                    with gr.Column():
                        interval_slider = gr.Slider(
                            minimum=3, maximum=15, value=5, step=1,
                            label="Description Interval (seconds)",
                            info="How often to announce scene changes"
                        )
                        start_btn = gr.Button("▶️ Start Live Stream", variant="primary", size="lg")
                        stop_btn = gr.Button("⏹️ Stop Live Stream", variant="stop", size="lg")
                        refresh_btn = gr.Button("🔄 Refresh Feed", size="sm")
                        stream_status = gr.Textbox(label="Status", interactive=False)
                    
                    with gr.Column():
                        live_video = gr.Image(label="Live Feed")
                        gr.Markdown("""
                        **How to use:**
                        1. Click "Start Live Stream"
                        2. The system will automatically describe what it sees
                        3. Click "Refresh Feed" to update the video display
                        4. Scene descriptions are spoken aloud when changes are detected
                        5. Click "Stop" when done
                        """)
                
                start_btn.click(
                    fn=start_live_stream,
                    inputs=[interval_slider],
                    outputs=[stream_status]
                )
                
                stop_btn.click(
                    fn=stop_live_stream,
                    outputs=[stream_status]
                )
                
                # Manual refresh for live video
                refresh_btn.click(
                    fn=get_live_frame,
                    outputs=[live_video]
                )
        
        # Help Tab
        with gr.Tab("ℹ️ Help & Tips"):
            gr.Markdown("""
            ## How to Use This Assistant
            
            ### For Visually Impaired Users:
            1. **Webcam Mode**: Click "webcam" to capture your environment in real-time
            2. **Audio Feedback**: Keep audio enabled to hear descriptions
            3. **Quick Analysis**: Use preset modes for specific tasks
            4. **AI Assistant**: Ask natural language questions about your surroundings
            
            ### Available Modes:
            - **Object Detection**: Identifies items in your environment
            - **Face Detection**: Counts people in frame
            - **QR/Barcode Scanner**: Reads product codes and QR codes
            - **Text Reading**: Reads signs, labels, and documents
            - **Color Identification**: Identifies dominant colors
            - **Complete Analysis**: Full scene understanding
            
            ### Tips for Best Results:
            - 💡 Hold objects 1-2 feet from camera
            - 📏 Ensure good lighting
            - 🎯 Keep camera steady for text reading
            - 🔊 Use headphones for privacy
            - 🤖 Use AI Assistant for complex questions
            
            ### Keyboard Shortcuts:
            - Press `Tab` to navigate between fields
            - Press `Enter` to activate buttons
            - Press `Space` to toggle checkboxes
            
            ### Technical Requirements:
            - Working webcam for real-time capture
            - Internet connection for AI features
            - Microphone for voice features (optional)
            """)
    
    return demo


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_gui() -> None:
    """Launch the GUI application."""
    try:
        demo = create_interface()
        demo.launch(
            share=False,
            inbrowser=True,
            show_error=True,
            server_port=7860
        )
    except Exception as exc:
        logger.error("Failed to launch GUI: %s", exc)
        raise


if __name__ == "__main__":
    run_gui()
