"""
Accessibility Vision Assistant - Main Application
Production-grade visual assistance system for visually impaired users

Features:
- Real-time object detection
- Face detection and counting
- QR/Barcode scanning
- OCR (text reading)
- Color identification
- AI-powered scene understanding
- Audio feedback

Optimized for low-compute systems.
"""
import cv2
import logging
from vision.detector import ObjectDetector
from speech.tts import speak, speak_async
from speech.stt import transcribe_audio
from nlp.prompt_engine import get_contextual_response
from gui.interface import run_gui

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def console_mode():
    """Voice-controlled console mode for accessibility"""
    try:
        detector = ObjectDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            logger.error("Failed to open camera")
            speak("Failed to open camera. Please check your webcam connection.")
            return
        
        # Set lower resolution for low-compute systems
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        speak("Accessibility assistant started. Press Enter to capture and analyze.")
        
        while True:
            try:
                user_input = input("\n[Press Enter to start voice command, or 'q' to quit]: ")
                
                if user_input.lower() == 'q':
                    speak("Goodbye!")
                    break
                
                # Get voice command
                speak("What would you like to know?")
                query = transcribe_audio(duration=5)
                
                if not query:
                    speak("I didn't catch that. Please try again.")
                    continue
                
                logger.info(f"User query: {query}")
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    speak("Failed to capture image. Please try again.")
                    continue
                
                # Detect objects
                annotated, objects = detector.detect(frame)
                
                # Show annotated frame
                cv2.imshow("Vision Assistant", annotated)
                cv2.waitKey(1)
                
                # Get AI response
                response = get_contextual_response(query, objects)
                logger.info(f"Response: {response}")
                
                # Speak response
                speak(response)
                
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                speak("An error occurred. Please try again.")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        logger.error(f"Fatal error in console mode: {e}")
        speak("A fatal error occurred. Please restart the application.")


def main():
    """Main entry point"""
    import sys
    
    print("=" * 60)
    print("   ACCESSIBILITY VISION ASSISTANT")
    print("   Empowering visually impaired users")
    print("=" * 60)
    print()
    print("Choose mode:")
    print("1. GUI Mode (Recommended) - Visual interface with webcam")
    print("2. Console Mode - Voice-controlled terminal interface")
    print("3. Exit")
    print()
    
    try:
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            logger.info("Starting GUI mode...")
            run_gui()
        elif choice == "2":
            logger.info("Starting console mode...")
            console_mode()
        elif choice == "3":
            print("Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Starting GUI mode by default...")
            run_gui()
            
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
