import gradio as gr
import cv2
from vision.detector import ObjectDetector
from vision.face_recognition import detect_faces
from vision.barcode_reader import detect_barcodes
from vision.ocr import extract_text
from vision.color_detector import detect_dominant_color
from vision.clip_vqa import answer_question
from speech.tts import speak

# Initialize models
yolo_detector = ObjectDetector()

def process_image(image, mode, question):
    results = {}
    
    if mode == "Object Detection":
        frame, labels = yolo_detector.detect(image)
        results["description"] = f"Detected objects: {', '.join(labels)}"
    #elif mode == "Face Recognition":
     #   frame, count = detect_faces(image)
      #  results["description"] = f"Detected {count} face(s)"
    elif mode == "QR/Barcode Detection":
        frame, codes = detect_barcodes(image)
        results["description"] = f"Codes: {', '.join(codes)}"
    elif mode == "OCR (Text Reading)":
        frame, text = extract_text(image)
        results["description"] = f"Text: {text}"
    elif mode == "Dominant Color Detection":
        frame, color = detect_dominant_color(image)
        results["description"] = f"Dominant color: {color}"
    elif mode == "CLIP Q&A":
        frame, answer = answer_question(image, question)
        results["description"] = f"Answer: {answer}"
    else:
        frame = image
        results["description"] = "Unknown mode selected."

    speak(results["description"])
    return frame, results["description"]

# Interface
def run_gui():
    gr.Interface(
        fn=process_image,
        inputs=[
            gr.Image(type="numpy", label="Input Frame"),
            gr.Radio(["Object Detection", "Face Recognition", "QR/Barcode Detection", 
                      "OCR (Text Reading)", "Dominant Color Detection", "CLIP Q&A"],
                     label="Select Mode"),
            gr.Textbox(label="Optional Question (for CLIP only)", value="")
        ],
        outputs=[
            gr.Image(type="numpy", label="Processed Frame"),
            gr.Textbox(label="Assistant's Description")
        ],
        live=False,
        title="Smart Visual Assistant",
        description="Select a mode, upload an image or use webcam to see real-time detection + assistant reply."
    ).launch()

