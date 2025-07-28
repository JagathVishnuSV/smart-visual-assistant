import cv2
from pyzbar.pyzbar import decode

def detect_barcodes(image):
    barcodes = decode(image)
    codes = []

    for barcode in barcodes:
        x, y, w, h = barcode.rect
        text = barcode.data.decode("utf-8")
        codes.append(text)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    return image, codes
