import cv2
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import numpy as np

def send_coords(params):
  # interface with arduino
  return 2

def coords_frame_average(coords, num_frames):
  ave_coords = []

  for i, idx in enumerate(coords):
    for j, jdx in enumerate(idx):
      if len(ave_coords) < j+1:
        ave_coords.append([jdx[0], jdx[1], jdx[2]])
      else:
        ave_coords[j][1] += jdx[1]
        ave_coords[j][2] += jdx[2] 
  
  for i, idx in enumerate(ave_coords):
    ave_coords[i][1] = ave_coords[i][1] / num_frames
    ave_coords[i][2] = ave_coords[i][2] / num_frames
  
  return ave_coords

def calculate_parameters(point_denorm):
  #point_denorm: [num. of point][num. of axis]
  #num. of axis: 1 --> x, 2 --> y, 3 --> z
  
  bottom_lip = np.sqrt((point_denorm[bottom_lip_index[0]][1] - point_denorm[bottom_lip_index[1]][1])**2 + (point_denorm[bottom_lip_index[0]][2] - point_denorm[bottom_lip_index[1]][2])**2)
  top_lip = np.sqrt((point_denorm[top_lip_index[0]][1] - point_denorm[top_lip_index[1]][1])**2 + (point_denorm[top_lip_index[0]][2] - point_denorm[top_lip_index[1]][2])**2)

  mouth_ratio = np.sqrt((point_denorm[mouth_ratio_index[0]][1] - point_denorm[mouth_ratio_index[1]][1])**2 + (point_denorm[mouth_ratio_index[0]][2] - point_denorm[mouth_ratio_index[1]][2])**2)
  nose_gap = np.sqrt((point_denorm[nose_gap_index[0]][1] - point_denorm[nose_gap_index[1]][1])**2 + (point_denorm[nose_gap_index[0]][2] - point_denorm[nose_gap_index[1]][2])**2)

  top_eyebrow_l_ratio = np.sqrt((point_denorm[top_eyebrow_l_ratio_index[0]][1] - point_denorm[top_eyebrow_l_ratio_index[1]][1])**2 
                                + (point_denorm[top_eyebrow_l_ratio_index[0]][2] - point_denorm[top_eyebrow_l_ratio_index[1]][2])**2)

  if mouth_ratio > nose_gap:
    print('Mouth really open: {:.2f}'.format(mouth_ratio))
  elif mouth_ratio > (bottom_lip + top_lip):
    print('Mouth open: {:.2f}'.format(mouth_ratio))
  
  if top_eyebrow_l_ratio < nose_gap:
    print('Ancelotti mode: {:.2f}'.format(top_eyebrow_l_ratio))
  # TO MEET CERTAIN THERESHOLDS, COMPARE BETWEEN DIFFERENT POINTS, LIKE LIP THICKNESS OR DISTANCE BETWEEN EYES
  return 0


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

image_w = 2880  #change according to computer
image_h = 1800  #use tkinter???
color = [0, 0, 0]
selected_points = [0, 12, 13, 14, 15, 17, 1, 5, 107, 108, 336, 337] #points of face model of mediapipe, check reference image
selected_points.sort()  #list must be sorted

#bottom_lip   --> 15, 17
#top_lip      --> 0, 12
#mouth ratio  --> 13, 14
#nose_gap     --> 1, 5
#top_eyebrow_l_ratio  --> 107, 108
#top_eyebrow_r_ratio  --> 336, 337
bottom_lip_index = []
top_lip_index = []
mouth_ratio_index = []
nose_gap_index = []
top_eyebrow_l_ratio_index = []
top_eyebrow_r_ratio_index = []

for i, idx in enumerate(selected_points):
  print(i, idx)
  if idx == 0:
    top_lip_index.append(i)
  elif idx == 12:
    top_lip_index.append(i)
  elif idx == 15:
    bottom_lip_index.append(i)
  elif idx == 17:
    bottom_lip_index.append(i)
  elif idx == 1:
    nose_gap_index.append(i)
  elif idx == 5:
    nose_gap_index.append(i)
  elif idx == 13:
    mouth_ratio_index.append(i)
  elif idx == 14:
    mouth_ratio_index.append(i)
  elif idx == 107:
    top_eyebrow_l_ratio_index.append(i)
  elif idx == 108:
    top_eyebrow_l_ratio_index.append(i)
  elif idx == 336:
    top_eyebrow_r_ratio_index.append(i)
  elif idx == 337:
    top_eyebrow_r_ratio_index.append(i)


denormalized_coords = []
coord_average = []
num_frame_average = 3
k = 0

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
        '''mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          landmark_drawing_spec=DrawingSpec(color=(0, 0, 0), thickness=1, circle_radius=1))'''
        mp_drawing.draw_landmarks(
          image=image,
          landmark_list=face_landmarks,
          connections=mp_face_mesh.FACEMESH_TESSELATION,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp_drawing_styles
          .get_default_face_mesh_tesselation_style())
        
        j = 0  #selected_points selection
        
        for i in range(len(results.multi_face_landmarks[0].landmark)):
          if i == selected_points[j]:            
            coord_average.append([selected_points[j], round(results.multi_face_landmarks[0].landmark[i].x * image_w), 
                                        round(results.multi_face_landmarks[0].landmark[i].y * image_h)])

            j += 1
            
            if j == len(selected_points):  # fill the whole buffer to calculate ratios
              j = 0               
              denormalized_coords.append(coord_average)
              k += 1

              if k == num_frame_average:
                k = 0
                calculate_parameters(coords_frame_average(denormalized_coords, num_frame_average))
                denormalized_coords.clear()

              coord_average.clear()


    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
