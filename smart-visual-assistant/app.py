from vision.detector import ObjectDetector
from speech.tts import speak
from speech.stt import transcribe_audio
from nlp.prompt_engine import build_prompt, dummy_llm_response
import cv2
from gui.interface import run_gui
def main():
    detector = ObjectDetector()
    cap = cv2.VideoCapture(0)

    while True:
        input("Press Enter to start voice command...")
        query = transcribe_audio()
        print(f"User query: {query}")

        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            continue

        annotated,objects = detector.detect(frame)
        cv2.imshow("Vision",annotated)
        cv2.waitKey(1)

        prompt = build_prompt(query,objects)
        print(f"[PROMPT]: {prompt}")
        response = dummy_llm_response(prompt)
        print(f"[RESPONSE]: {response}")

        speak(response)
        
        if input("Type 'q' to quit or Enter to continue: ").lower() == 'q':
            break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    #main()
    run_gui()
