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
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']  # Emotions to be predicted
except Exception as e:
    print(f"Error loading cascade classifiers: {e}")
    exit()

def detect_emotion(frame, image_path):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # Adjust the parameters as needed

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
        
        result_dict = {
            'Test date': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'File name': image_path,
            'Angry': f"{round(prediction[emotion_labels.index('Angry')] * 100, 2)}%",
            'Disgust': f"{round(prediction[emotion_labels.index('Disgust')] * 100, 2)}%",
            'Fear': f"{round(prediction[emotion_labels.index('Fear')] * 100, 2)}%",
            'Happy': f"{round(prediction[emotion_labels.index('Happy')] * 100, 2)}%",
            'Neutral': f"{round(prediction[emotion_labels.index('Neutral')] * 100, 2)}%",
            'Sad': f"{round(prediction[emotion_labels.index('Sad')] * 100, 2)}%",
            'Surprise': f"{round(prediction[emotion_labels.index('Surprise')] * 100, 2)}%",
            'Faces': faces
        }

    return result_dict, frame

def save_to_excel(result_list, output_path='../Result/result_image.xlsx', save_predicted_images=True):
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
    
    # Lưu ảnh dự đoán vào thư mục "predict"
    if save_predicted_images:
        predict_directory = os.path.join(os.path.dirname(output_path), '../Predict')
        if not os.path.exists(predict_directory):
            os.makedirs(predict_directory)

        for result in result_list:
            image_path = result['File name']
            image = cv2.imread(image_path)
            faces = result['Faces'] # Lấy giá trị faces từ result_dict

            for (x, y, w, h) in faces:
                # Vẽ khung bao xung quanh khuôn mặt
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Tạo tên file mới dựa trên thời gian và tên file cũ
                timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                _, file_extension = os.path.splitext(os.path.basename(image_path))
                new_filename = f"predict_{timestamp}{file_extension}"
                new_file_path = os.path.join(predict_directory, new_filename)

                # Tạo một frame mới để vẽ kết quả
                frame = image.copy()

                # Vẽ kết quả lên frame
                main_label = result['Angry'].split()[0]
                main_label_position = (x, y - 10)
                cv2.putText(frame, main_label, main_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                other_labels = [label for label in emotion_labels if label != main_label]
                for i, other_label in enumerate(other_labels):
                    # Chuyển đổi giá trị thành số và sau đó làm tròn
                    other_confidence = round(float(result[other_label].rstrip('%')) * 100, 2)
                    other_label_text = f"{other_label}: {other_confidence}%"
                    other_label_position = (x, y + (i + 1) * 20)
                    cv2.putText(frame, other_label_text, other_label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Lưu ảnh dự đoán
                cv2.imwrite(new_file_path, frame)

    print(f"Predicted images saved to {predict_directory}")

def test_on_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Couldn't read the image at path {image_path}.")
        return

    result_dict, result_frame = detect_emotion(image, image_path)
    
    # Lưu kết quả vào danh sách
    result_list = [result_dict]
    
    # Lưu danh sách kết quả vào Excel
    save_to_excel(result_list)
    
    cv2.imshow("Emotion Recognition", result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Sử dụng argparse để thêm đối số dòng lệnh
    parser = argparse.ArgumentParser(description='Emotion Recognition from Image')
    parser.add_argument('-i', '--image', dest='image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    test_on_image(args.image_path)
