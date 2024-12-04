import cv2  #Loads the OpenCV library, which is used for video capture, image processing, and drawing on images (e.g., rectangles for faces)
from deepface import DeepFace #Imports DeepFace, a Python library for facial recognition and emotion analysis.
import pandas as pd #handling structured data like creating and saving files
from collections import defaultdict
import datetime
import os
import sys
import google.generativeai as genai  # Added import for Gemini

# Add the script directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
#Determines the directory of the current script file.
sys.path.append(script_dir)
#Determines the directory of the current script file.

# Import the AI Interview Emotion Report Generator
from AIgeneration import AIInterviewEmotionReportGenerator 
#Imports a custom module/class for generating AI interview emotion reports.

# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Haarcascade for face detection
face_classifier = cv2.CascadeClassifier(r'FER/haarcascade_frontalface_default.xml')

# Google API Key (replace with your actual key or environment variable)
API_KEY = 'AIzaSyBgZm8E3GFSX5RpHG4TzItfYFUOAK40ByM'

cap = cv2.VideoCapture(0)  # Start video capture

# Emotion labels (for counting occurrences)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Initialize a dictionary to store the count of each emotion
emotion_count = defaultdict(int)

# Create a pandas DataFrame for the report (continuous emotion tracking)
columns = ['Start Time', 'End Time', 'Emotion', 'Probability']
report_df = pd.DataFrame(columns=columns)

# Tracking variables for continuous emotion detection
current_emotion = None
emotion_start_time = None
is_recording = False

# Variable to store the most recent CSV filename for AI report generation
recent_csv_filename = None

def generate_ai_report():
    """
    Generate AI-powered interview emotion analysis report
    """
    global recent_csv_filename
    if recent_csv_filename and os.path.exists(recent_csv_filename):
        try:
            # Initialize report generator
            report_generator = AIInterviewEmotionReportGenerator(recent_csv_filename, API_KEY)
            
            # Generate and save report with default naming
            report_generator.save_report()
            print(f"AI Report generated for {recent_csv_filename}")
        except Exception as e:
            print(f"Error generating AI report: {e}")
    else:
        print("No recent emotion report found. Start recording first.")

while True:
    _, frame = cap.read()  # Read each frame from the camera
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    faces = face_classifier.detectMultiScale(gray)  # Detect faces in the frame

    detected_emotion = None
    detected_probability = 0

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw a rectangle around detected faces
        roi_color = frame[y:y+h, x:x+w]  # Extract region of interest (ROI) in color

        try:
            # Use DeepFace emotion analysis with VGG model
            analysis = DeepFace.analyze(
                roi_color, 
                actions=['emotion'], 
                enforce_detection=False
            )

            # Handle results (DeepFace returns a list for multiple faces)
            if isinstance(analysis, list):
                analysis = analysis[0]  # Get the first face's analysis

            # Extract the dominant emotion
            detected_emotion = analysis['dominant_emotion']
            detected_probability = analysis['emotion'][detected_emotion]

            # Increment emotion count
            emotion_count[detected_emotion] += 1

            # Put the emotion label with its probability on the frame
            label_with_percentage = f"{detected_emotion} ({detected_probability:.2f}%)"
            label_position = (x, y)
            cv2.putText(frame, label_with_percentage, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        except Exception as e:
            # Handle errors gracefully
            print(f"Error: {e}")
            cv2.putText(frame, "Error Detecting Emotion", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Emotion change tracking logic
    current_time = datetime.datetime.now()

    if is_recording:
        # If no face detected or emotion changed, log the previous emotion
        if not detected_emotion or (current_emotion and detected_emotion != current_emotion):
            new_row = pd.DataFrame({
                'Start Time': [emotion_start_time],
                'End Time': [current_time],
                'Emotion': [current_emotion],
                'Probability': [detected_probability]
            })
            report_df = pd.concat([report_df, new_row], ignore_index=True)
            
            # Reset current emotion tracking
            current_emotion = None
            emotion_start_time = None

    # Start tracking new emotion
    if is_recording and detected_emotion and not current_emotion:
        current_emotion = detected_emotion
        emotion_start_time = current_time

    # Display recording status
    if is_recording:
        cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the updated frame
    cv2.imshow('Emotion Detector', frame)

    # Display emotion counts on the frame
    report_text = "\n".join([f"{emotion}: {count}" for emotion, count in emotion_count.items()])
    cv2.putText(frame, f"Emotion Counts:\n{report_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Keyboard commands
    key = cv2.waitKey(1) & 0xFF
    
    # Start recording with 'r' key
    if key == ord('r'):
        is_recording = True
        report_df = pd.DataFrame(columns=columns)  # Reset report
        current_emotion = None
        emotion_start_time = None
        print("Recording started...")

    # Stop recording and save with 'p' key
    elif key == ord('p'):
        if is_recording:
            is_recording = False
            # If there's an ongoing emotion when stopped, log it
            if current_emotion:
                new_row = pd.DataFrame({
                    'Start Time': [emotion_start_time],
                    'End Time': [datetime.datetime.now()],
                    'Emotion': [current_emotion],
                    'Probability': [detected_probability]
                })
                report_df = pd.concat([report_df, new_row], ignore_index=True)

            # Generate a unique filename using timestamp
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            recent_csv_filename = os.path.join(SCRIPT_DIR, f'emotion_report_{timestamp}.csv')
            report_df.to_csv(recent_csv_filename, index=False)
            print(f"Report saved to '{recent_csv_filename}'")

            # Reset tracking
            current_emotion = None
            emotion_start_time = None

    # Generate AI Report with 'a' key
    elif key == ord('a'):
        generate_ai_report()

    # Exit when 'q' is pressed
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()