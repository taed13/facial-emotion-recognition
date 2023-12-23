#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Loading pre-trained parameters for the cascade classifier
try:
    face_classifier = cv2.CascadeClassifier('/Users/letiendat/Documents/Semes-Six/TGMT/Facial-Emotion-Recognition/Model/haarcascade_frontalface_default.xml')  # Face Detection
    classifier = load_model('/Users/letiendat/Documents/Semes-Six/TGMT/Facial-Emotion-Recognition/Model/model.h5')  # Load model
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Emotions to be predicted
except Exception as e:
    print(f"Error loading cascade classifiers: {e}")
    exit()

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # Adjust the parameters as needed

    for (x, y, w, h) in faces:
        # Extract the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        prediction = classifier.predict(roi)[0]

        # Display the main emotion
        main_label = emotion_labels[prediction.argmax()]
        main_confidence = round(max(prediction) * 100, 2)
        main_label_text = f"{main_label} ({main_confidence}%)"
        main_label_position = (x, y - 10)  # Adjust the position to be above the rectangle
        cv2.putText(frame, main_label_text, main_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display other emotions with percentages
        other_labels = [label for label in emotion_labels if label != main_label]
        for i, other_label in enumerate(other_labels):
            other_confidence = round(prediction[emotion_labels.index(other_label)] * 100, 2)
            other_label_text = f"{other_label}: {other_confidence}%"
            other_label_position = (x, y + (i + 1) * 20)  # Adjust position for each label
            cv2.putText(frame, other_label_text, other_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw the rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return frame

def test_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    

    if not cap.isOpened():
        print(f"Error: Could not open the video at path {video_path}.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        frame = detect_emotion(frame)

        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace 'path/to/your/video.mp4' with the actual path to your test video
    test_video_path = '/Users/letiendat/Documents/Semes-Six/TGMT/Facial-Emotion-Recognition/Video/video_test.mp4'
    test_on_video(test_video_path)
