# Computer Vision and Speech Interaction
This project use computer vision to monitor a user's behavior through a webcam feed. The system detects the presence of people and mobile phones, tracks head pose, and monitors mouth openness to identify speech activity.

# Requirements
Make sure to install the required Python packages before running the code:


# Features
# Object Detection
The system uses YOLOv8 for real-time object detection. It identifies and counts the number of people and detects mobile phones in the webcam feed.

# Head Pose Detection
Facial landmarks are detected using MediaPipe Face Mesh. The system calculates head pose angles and provides feedback if the user's head is tilted in any direction.

# Mouth Openness Detection
The system monitors mouth openness to detect speech activity. If the user keeps their mouth closed for a specified time threshold, the system prompts them to speak.


# User Interaction
The user is informed about their head pose, mouth openness status, and warnings related to the presence of multiple people or a mobile phone. The system prompts the user to speak when appropriate.

# Notes
Press 'q' to exit the program.
The system stops if a mobile phone is detected, or if the user's head pose is consistently incorrect for an extended period.
Feel free to customize the code and experiment with different models and parameters based on your requirements.






