import cv2
import mediapipe as mp
import time
import numpy as np
from ultralytics import YOLO
import threading
import pyttsx3
import whisper
from transformers import pipeline
from threading import Thread
from queue import Queue
from tempfile import NamedTemporaryFile
import speech_recognition as sr
import time
import keyboard


# # Varables
# transcription=[]
# r = sr.Recognizer()
# temp_file = NamedTemporaryFile(delete=True).name
# audio_queue = Queue()




classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
              "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
              "hot dog", "pizza", "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet",
              "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
              "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# loading model
model = YOLO('yolov8s.pt')

# defining pipeleine for whisper-tiny model
pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny"
)


def detecting_mobile_and_people_count(img, start_time1=0,time1=0):
    results = model(img, stream=True)
    people_count = 0
    mobile_detected = False
    
  

    # iterating through the result
    for r in results:
        boxes = r.boxes
        # iterating through every detection one by one
        for box in boxes:

            cls = int(box.cls[0])  # changing the class number from tensor to integer
            label = classNames[cls]  # retrieving the class name
            conf_score = int(box.conf[0] * 100)

            # Checking the labels if a person or cell phone has been detected,
            # if a person is detected then counting the number of people

            if label == 'person' and conf_score > 60:
                people_count += 1
            elif label == 'cell phone':
                mobile_detected = True
                if not start_time1:
                    start_time1 = time.time()
                
                

        # checking if there is more than one person in the fame, then show the error
        if people_count > 1:
            cv2.putText(img, "Warning: More than one person", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2)
            
        if people_count == 0:
            cv2.putText(img, f"Face not detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
           
            
    if mobile_detected:
        
        time1 =time.time() - start_time1
        cv2.putText(img, f" Warning: Mobile Phone detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif mobile_detected==False:
        start_time1=0
        time1=0
        
    return img, start_time1, time1


# variables
frame_counter = 0

# constants
FONTS = cv2.FONT_HERSHEY_COMPLEX

map_face_mesh = mp.solutions.face_mesh


# Landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # List of (x, y) coordinates
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # Return the list of tuples for each landmark
    return mesh_coord


# Euclidean distance
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


# Pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    # Count black pixels in each part
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)
    # Create a list of these values
    eye_parts = [right_part, center_part, left_part]

    # Get the index of the max value in the list
    max_index = eye_parts.index(max(eye_parts))
    pos_eye = ''
    color=(255,0,0)
    if max_index == 0:
        pos_eye = "Look straight"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 2:
        pos_eye = 'Look straight'
        color = [utils.BLACK, utils.GREEN]
    return pos_eye, color


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


mouth_open_threshold = 1
mouth_closed_time_threshold = 10


def vision_worker():
    flag =0
    mouth_closed_start_time = None
    cap = cv2.VideoCapture(0)
    start_time1,time1=0, 0
    COUNT = 0

    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:

            # frame_counter += 1
            ret, frame = cap.read()

            if not ret:
                break

            img_after_detection ,start_time1,time1 = detecting_mobile_and_people_count(frame,start_time1,time1)
            time1 = int(time1)
            
            
            
            

            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False

            results = face_mesh.process(frame_rgb)
            result = face_mesh.process(image)

            image.flags.writeable = True

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_h, img_w, img_c = image.shape
            face_3d = []
            face_2d = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                            x, y = int(lm.x * img_w), int(lm.y * img_h)

                            # Get the 2D Coordinates
                            face_2d.append([x, y])

                            # Get the 3D Coordinates
                            face_3d.append([x, y, lm.z])

                            # Convert it to the NumPy array
                    face_2d = np.array(face_2d, dtype=np.float64)

                    # Convert it to the NumPy array
                    face_3d = np.array(face_3d, dtype=np.float64)

                    # The camera matrix
                    focal_length = 1 * img_w

                    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                           [0, focal_length, img_w / 2],
                                           [0, 0, 1]])

                    # The Distance Matrix
                    dist_matrix = np.zeros((4, 1), dtype=np.float64)

                    # Solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                    # Get rotational matrix
                    rmat, jac = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360

                    # See where the user's head is tilting
                    if y < -10:
                        head_pose_text = "PLEASE LOOK STRAIGHT"
                        COUNT+= 1
                        
                    elif y > 10:
                        head_pose_text = "PLEASE LOOK STRAIGHT"
                        COUNT+= 1
                
                    elif x > 15:
                        head_pose_text = "PLEASE LOOK STRAIGHT"
                        COUNT+=1
                       
                    elif x < -10:
                        head_pose_text = "PLEASE LOOK STRAIGHT"
                        COUNT+= 1
                    
                    else:
                        head_pose_text = "  "
                        COUNT = 0

                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0]), int(nose_2d[1]))
                    p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))

                    cv2.line(image, p1, p2, (255, 0, 0), 2)

                    # Add the text on the image
                    cv2.putText(image, head_pose_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(frame, head_pose_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    # Extract landmarks for the upper and lower lips
                    upper_lip_landmarks = landmarks.landmark[0]
                    lower_lip_landmarks = landmarks.landmark[87]

                    # Calculate the distance between the upper and lower lip landmarks
                    lip_distance = abs(upper_lip_landmarks.y - lower_lip_landmarks.y) * 100
                    # print(int(lip_distance))

                    mouth_open = lip_distance > 2
                    
                    if mouth_open:
                        mouth_closed_start_time = None
                    else:
                        if mouth_closed_start_time is None:
                            
                            mouth_closed_start_time = time.time()
                            
                        else:
                            elapsed_time = time.time() - mouth_closed_start_time
                            
                            if elapsed_time > mouth_closed_time_threshold:
                                flag = 1
                                break

                    # Display the mouth openness status
                    status_text = " " if mouth_open else "PLEASE SPEAK"
                    
                    

                cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            if flag==1 or time1==10 or COUNT>20:
                break
                
            else:
                cv2.imshow('Combined Detection', frame)
                


            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    


# Create a thread for webcam-based computer vision
webcam_thread = threading.Thread(target=vision_worker)
webcam_thread.daemon = True
webcam_thread.start()