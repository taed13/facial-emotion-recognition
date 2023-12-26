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

        results_list.append(result_dict)

        # Draw the rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return results_list, frame, faces

def save_video_results_to_excel(result_list, output_path='../Result/result_video.xlsx'):
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

def save_predicted_images(predict_directory, faces, result_list, frame):
    for i, ((x, y, w, h), result) in enumerate(zip(faces, result_list)):
        image = frame.copy()

        # Kiểm tra xem có khuôn mặt được phát hiện không
        if len(faces) > 0:
            # Vòng lặp duyệt qua tất cả các khuôn mặt được phát hiện
            for j, (x, y, w, h) in enumerate(faces):
                # Vẽ khung bao xung quanh khuôn mặt
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Kiểm tra xem có kết quả cảm xúc cho khuôn mặt này không
                if 'Main Label' in result:
                    # Lấy label của cảm xúc có tỉ lệ nhận dạng cao nhất từ result_dict
                    main_label = result['Main Label']
                    main_confidence = result['Main Confidence']

                    # Tạo tên file mới dựa trên thời gian và tên file cũ
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    new_filename = f"predict_{timestamp}.png"  # Sử dụng phần mở rộng .png
                    new_file_path = os.path.join(predict_directory, new_filename)

                    # Vẽ kết quả lên frame
                    main_label_position = (x, y - 10)
                    cv2.putText(frame, f"{main_label} ({main_confidence}%)", main_label_position,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                    other_labels = [label for label in emotion_labels if label != main_label]
                    for k, other_label in enumerate(other_labels):
                        # Loại bỏ ký tự '%' và sau đó chuyển đổi thành số float
                        other_confidence_str = result[other_label].rstrip('%')
                        other_confidence = round(float(other_confidence_str), 2)
                        other_label_text = f"{other_label}: {other_confidence}%"
                        other_label_position = (x, y + (k + 1) * 20)
                        cv2.putText(frame, other_label_text, other_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Lưu ảnh dự đoán
                    cv2.imwrite(new_file_path, frame)

    print(f"Predicted video saved to {predict_directory}")

def test_on_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open the video at path {video_path}.")
        return

    results_list = []  # List to store all results
    
    # Tạo đường dẫn cho thư mục Predict_Video
    output_video_dir = os.path.join(os.path.dirname(video_path), '../Predict/Predict_Video')
    
    # Kiểm tra và tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_video_dir):
        os.makedirs(output_video_dir)
        print(f"Created directory: {output_video_dir}")
    
    # Get the frame size
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        cap.release()
        return

    frame_height, frame_width, _ = frame.shape
    
    # Tạo một đối tượng VideoWriter với tên file có mã thời gian
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    output_video_filename = f'predicted_video_{timestamp}.mp4'
    output_video_path = os.path.join(output_video_dir, output_video_filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Có thể sử dụng 'XVID' hoặc 'MJPG' tùy thuộc vào hệ thống
    output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width, frame_height))
    
    if not output_video.isOpened():
        print(f"Error: Could not create the output video file at path {output_video_path}.")
        cap.release()
        return

    print(f"Creating predicted video at {output_video_path}")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("End of video.")
            break

        # Detect emotions in the current frame
        current_results_list, result_frame, faces = detect_emotion(frame)

        # Save the results to the overall list
        results_list.extend(current_results_list)

        # Display the frame with emotions detected
        cv2.imshow("Emotion Recognition", result_frame)
        output_video.write(result_frame)
        
        # Check if the user wants to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the results to an Excel file
    save_video_results_to_excel(results_list, output_path='../Result/result_video.xlsx')

    # Giải phóng tài nguyên VideoWriter
    output_video.release()
    print(f"Predicted video saved at {os.path.abspath(output_video_path)}")

if __name__ == "__main__":
    # Sử dụng argparse để thêm đối số dòng lệnh
    parser = argparse.ArgumentParser(description='Emotion Recognition from Video')
    parser.add_argument('-vid', '--video', dest='video_path', type=str, help='Path to the video file')
    args = parser.parse_args()
    
    test_on_video(args.video_path)
