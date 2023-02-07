import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import numpy as np

def send_coords(params):
  # interface with arduino
  return 2

def get_selected_points_coords(landmark_list, selected_points):
  return 1

def calculate_parameters(point_denorm):
  #point_denorm: [num. of point][num. of axis]
  #num. of axis: 1 --> x, 2 --> y
  scalar = 2
  relative_meas = np.sqrt((point_denorm[3][2] - point_denorm[2][2])**2 + (point_denorm[3][1] - point_denorm[2][1])**2)
  mouth_ratio = np.sqrt((point_denorm[0][2] - point_denorm[1][2])**2 + (point_denorm[0][1] - point_denorm[1][1])**2)
  
  if mouth_ratio > 20:
    print('mouth ratio: ', mouth_ratio, 'meas: ', relative_meas, 'bora: ', mouth_ratio*scalar/relative_meas)
  
  
  return 0


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

image_w = 2880  #change according to computer
image_h = 1800  #use tkinter???
color = [0, 0, 0]
selected_points = [12, 15, 105, 107, 195, 197, 334] #points of face model of mediapipe, check reference image
selected_points.sort()  #list must be sorted

denormalized_coords = []


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

            j += 1
            
            if j == len(selected_points):  # fill the whole buffer to calculate ratios
              j = 0               
              calculate_parameters(denormalized_coords)                            
              denormalized_coords.clear()


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
