import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from time import sleep
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

image_w = 2880
image_h = 1800
color = [0, 0, 0]
selected_points = [13, 14, 105, 107, 334]
denormalized_coords = []
point_coords = []

def send_coords(params):
  return 2

def get_selected_points_coords(landmark_list, selected_points):
  return 1

def calculate_parameters(point_coords, point_denorm):
  mouth_ratio = np.sqrt((point_coords[0][1] - point_coords[1][1])**2 + (point_coords[0][0] - point_coords[1][0])**2)
  mouth_ratio_1 = np.sqrt((point_denorm[0][1] - point_denorm[1][1])**2 + (point_denorm[0][0] - point_denorm[1][0])**2)
  #print(mouth_ratio)
  print(mouth_ratio_1)
  return 0

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
cap = cv2.VideoCapture(0)
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          landmark_drawing_spec=DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1))
        
        j = 0  #selected_points selection
        for i in range(len(results.multi_face_landmarks[0].landmark)):
          if i == selected_points[j]:
            denormalized_coords.append([selected_points[j], round(results.multi_face_landmarks[0].landmark[i].x * image_w), 
                                        round(results.multi_face_landmarks[0].landmark[i].y * image_h)])
            point_coords.append([selected_points[j], results.multi_face_landmarks[0].landmark[i].x,
                             results.multi_face_landmarks[0].landmark[i].y, results.multi_face_landmarks[0].landmark[i].z])
            j += 1
            if j == 5:
              j = 0 
              calculate_parameters(point_coords, denormalized_coords)
              point_coords.clear()
              denormalized_coords.clear()
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
