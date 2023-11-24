from ultralytics import YOLO
import cv2
import time

#Downloading and loading the YOLO version 8 nano model
model=YOLO('yolov8s.pt')

# This is the list of objects that YOLO has been trained to detect. We will use this to find out the labels in the detection.
classNames=["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
             "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
             "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
             "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
             "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
             "hot dog", "pizza", "donut", "cake", "chain", "sofa", "pottedplant", "bed", "diningtable", "toilet",
             "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster",
             "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]




#Funtion to rescale the video

def rescale(img,scale=0.5):
    width=int(img.shape[1]*scale)
    height = int(img.shape[0] * scale)
    dimension=(width,height)
    return cv2.resize(img,dimension,interpolation=cv2.INTER_AREA)


# Funtion to detect cell phone and count the number of people in a frame.

def detecting_mobile_and_people_count(img,start_time1=0,start_time2=0,time1=0,time2=0):
    results = model(img, stream=True)
    people_count = 0

    #iterating through the result
    for r in results:
        boxes = r.boxes
        #iterating through every detection one by one
        for box in boxes:

            cls = int(box.cls[0])  # changing the class number from tensor to integer
            label = classNames[cls]  # retrieving the class name
            conf_score = int(box.conf[0] * 100)

            # Checking the labels if a person or cell phone has been detected,
            # if a person is detected then counting the number of people

            if label == 'person' and conf_score>60:
                people_count += 1
            elif label == 'cell phone':
                cv2.putText(img, "Mobile Phone has been detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # print("Mobile Phone has been detected")

        #checking if there is more than one person in the fame, then show the error
        if people_count > 1:
            cv2.putText(img, "Warning: There is more than one person", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # print("Warning: There is more than one person")

        #if no person is detected, then show "Face not detected"
        if people_count == 0:
            cv2.putText(img,"Face not detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # print("Face not detected")
    return img



cap=cv2.VideoCapture(0)
while True:
    success,img=cap.read()
    # img = rescale(img)    # Rescaling in case a video file has been given other than using web cam
    #calling the "detecting_mobile_and_people_count" function

    img_after_detection=detecting_mobile_and_people_count(img)
    cv2.imshow("Image", img_after_detection)

    #press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break