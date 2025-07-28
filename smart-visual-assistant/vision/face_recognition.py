import cv2
import face_recognition

def detect_faces(image):
    rgb_image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face.locations(rgb_image)

    for(top,right,bottom,left) in face_locations:
        cv2.rectangle(image,(left,top),(right,bottom),(0,255,0),2)
    
    return image, len(face_locations)