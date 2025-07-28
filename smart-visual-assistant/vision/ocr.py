import easyocr
import cv2

reader = easyocr.Reader(['en'])

def extract_text(image):
    results = reader.readtext(image)
    full_text = ""

    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(image, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        full_text += text + " "

    return image, full_text.strip()
