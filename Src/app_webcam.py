#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import pandas as pd
import os
from datetime import datetime
import argparse

# Loading pre-trained parameters for the cascade classifier
try:
    face_classifier = cv2.CascadeClassifier('../Model/haarcascade_frontalface_default.xml')  # Face Detection
    classifier_48 = load_model('../Model/model.h5')  # Load model
    classifier_96 = load_model('../Model/model_emotion_face_detection.h5')  # Load model
    classifier_299 = load_model('../Model/model_v1_inceptionV3.h5')  # Load model
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Emotions to be predicted
except Exception as e:
    print(f"Error loading cascade classifiers: {e}")
    exit()

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # Adjust the parameters as needed

    results_list = []  # List to store individual results

    for (x, y, w, h) in faces:
        # Extract the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        
        # =================================      Model Shape (48, 48, 1)      =================================
        # Resize the face ROI to match the input size of the model (48, 48, 1)
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        # Normalize the pixels to be in the range [0, 1]
        roi_color = roi_gray.astype('float') / 255.0
        roi_color = img_to_array(roi_color)
        
        # Expand dimensions to match the input shape of the new model
        roi_color = np.expand_dims(roi_color, axis=0)
        
        # Make prediction using the new model
        prediction = classifier_48.predict(roi_color)[0]
        
        # =================================      Model Shape (96, 96, 3)      =================================
        # # Resize the face ROI to match the input size of the model (96, 96, 3)
        # roi_gray = cv2.resize(roi_gray, (96, 96), interpolation=cv2.INTER_AREA)
        
        # # Convert the resized face ROI to RGB
        # roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB)
        
        # # Normalize the pixels to be in the range [0, 1]
        # roi_color = roi_color.astype('float') / 255.0
        
        # # Expand dimensions to match the input shape of the new model
        # roi_color = np.expand_dims(roi_color, axis=0)

        # # Make prediction using the new model
        # prediction = classifier_96.predict(roi_color)[0]
        
        # =================================      Model Shape (299, 299, 3)      =================================
        # # Resize the face ROI to match the input size of the model (299, 299, 3)
        # roi_gray = cv2.resize(roi_gray, (299, 299), interpolation=cv2.INTER_AREA)
        
        # # Convert the resized face ROI to RGB
        # roi_color = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2RGB)
        
        # # Normalize the pixels to be in the range [0, 1]
        # roi_color = roi_color.astype('float') / 255.0
        
        # # Expand dimensions to match the input shape of the new model
        # roi_color = np.expand_dims(roi_color, axis=0)

        # # Make prediction using the new model
        # prediction = classifier_299.predict(roi_color)[0]

        # Display the main emotion
        main_label = emotion_labels[prediction.argmax()]
        main_confidence = round(max(prediction) * 100, 2)

        # Display the main emotion for 48x48 model
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

        result_dict = {
            'Test date': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'Angry': f"{round(prediction[emotion_labels.index('Angry')] * 100, 2)}%",
            'Disgust': f"{round(prediction[emotion_labels.index('Disgust')] * 100, 2)}%",
            'Fear': f"{round(prediction[emotion_labels.index('Fear')] * 100, 2)}%",
            'Happy': f"{round(prediction[emotion_labels.index('Happy')] * 100, 2)}%",
            'Neutral': f"{round(prediction[emotion_labels.index('Neutral')] * 100, 2)}%",
            'Sad': f"{round(prediction[emotion_labels.index('Sad')] * 100, 2)}%",
            'Surprise': f"{round(prediction[emotion_labels.index('Surprise')] * 100, 2)}%",
            'Main Label': main_label,
            'Main Confidence': main_confidence,
        }
        
        results_list.append((result_dict, frame))

        # Draw the rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return results_list, frame

def save_webcam_results_to_excel(result_list, output_path='../Result/result_webcam.xlsx'):
    # Kiểm tra và tạo thư mục nếu chưa tồn tại
    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    # Nếu file Excel đã tồn tại, đọc nó và thêm kết quả mới
    if os.path.exists(output_path):
        existing_df = pd.read_excel(output_path)
        new_df = pd.DataFrame(result_list)
        df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # Nếu file Excel không tồn tại, tạo DataFrame mới từ danh sách kết quả
        df = pd.DataFrame(result_list)

    # Lưu DataFrame vào file Excel
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")
    
def save_webcam_video(result_list, output_dir='../Predict/Predict_Webcam'):
    # Kiểm tra và tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Lấy thông tin frame từ kích thước frame đầu tiên
    frame_height, frame_width, _ = result_list[0][1].shape

    # Tạo đối tượng VideoWriter với tên file có mã thời gian
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_filename = f'predicted_webcam_{timestamp}.mp4'
    output_path = os.path.join(output_dir, output_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Có thể sử dụng 'XVID' hoặc 'MJPG' tùy thuộc vào hệ thống
    output_video = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))

    if not output_video.isOpened():
        print(f"Error: Could not create the output video file at path {output_path}.")
        return

    print(f"Creating predicted video at {output_path}")

    # Lặp qua danh sách kết quả và ghi vào video
    for _, frame in result_list:
        output_video.write(frame)

    # Giải phóng tài nguyên VideoWriter
    output_video.release()
    print(f"Predicted video saved at {os.path.abspath(output_path)}")

def main():
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
        
    results_list = []  # List to store all results
    
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame from webcam.")
            break

        frame_height, frame_width, _ = frame.shape

        # Detect emotions in the current frame
        current_result, result_frame = detect_emotion(frame)
        
        # Save the frame with detected emotions to the result dictionary
        # current_result['frame'] = result_frame

        # Save the results to the overall list
        results_list.extend(current_result)

        # Display the frame with emotions detected
        cv2.imshow("Emotion Recognition", result_frame)

        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the results to an Excel file
    save_webcam_results_to_excel(results_list, output_path='../Result/result_webcam.xlsx')
    
    # Save the video with detected emotions
    save_webcam_video(results_list, output_dir='../Predict/Predict_Webcam')

if __name__ == "__main__":
    main()