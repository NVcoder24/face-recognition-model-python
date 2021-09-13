"""
This code should output if the faces in the pictures, 
the path to which the user points to match.
The code uses the computer vision library (opencv-python) 
and a trained neural model from the face_recognition library.
""""

import cv2
import face_recognition as fr

def get_img_rgb(path):
  return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def encode_face(img):
  return fr.face_encodings(img)[0]

def compare(face_x, face_y):
  return fr.compare_faces([face_x], face_y)[0]

face_x = encode_face(get_img_rgb(input("Path to image x: ")))
face_y = encode_face(get_img_rgb(input("Path to image y: ")))

print("Faces match!" if compare(face_x, face_y) else "Faces do not match!")
