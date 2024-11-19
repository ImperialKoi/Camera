import cv2
import easyocr
import numpy as np
import nltk
from nltk.corpus import words
from nltk.corpus import nps_chat
from openai import OpenAI

seen = []

client = OpenAI(api_key='') 

messages = [ {"role": "system", "content":  
              "You are a intelligent assistant."} ] 

posts = nps_chat.xml_posts()[:10000]

# Define the feature extraction function
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    features['has_solve_mark'] = 'solve' in post
    return features

# Create feature sets
featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]

# Split the data into training and test sets
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]

classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f'Accuracy: {accuracy:.4f}')

# Function to classify new input
def classify_input(text):
    features = dialogue_act_features(text)
    return classifier.classify(features)

# Initialize NLTK's English words corpus
english_words = set(words.words())

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

recognized_words = set()

def are_points_close(point1, point2, threshold=50):
    return abs(point1[0] - point2[0]) <= threshold and abs(point1[1] - point2[1]) <= threshold

def combine_bounding_boxes(box1, box2):
    x_coords = [point[0] for point in box1] + [point[0] for point in box2]
    y_coords = [point[1] for point in box1] + [point[1] for point in box2]
    return [
        [min(x_coords), min(y_coords)],
        [max(x_coords), min(y_coords)],
        [max(x_coords), max(y_coords)],
        [min(x_coords), max(y_coords)]
    ]

def merge_close_boxes(data, threshold=50):
    merged_data = []
    used = [False] * len(data)

    for i in range(len(data)):
        if used[i]:
            continue
        
        box1, text1, conf1 = data[i]
        new_box = box1
        new_text = text1
        
        for j in range(len(data)):
            if i == j or used[j]:
                continue

            box2, text2, conf2 = data[j]
            if any(are_points_close(p1, p2, threshold) for p1 in box1 for p2 in box2):
                new_box = combine_bounding_boxes(new_box, box2)
                new_text += " " + text2
                used[j] = True
        
        merged_data.append((new_box, new_text, conf1))
        used[i] = True
    
    return merged_data

# Function to perform OCR on an image and draw bounding boxes
def perform_ocr_and_draw_boxes(image):
    # Perform OCR
    result = reader.readtext(image)

    # Merge close boxes
    merged_data = merge_close_boxes(result)

    # Draw bounding boxes around the text
    for detection in result:
        # Extract bounding box coordinates and text
        bbox = detection[0]
        text = detection[1]

        # Check if the text is an English word
        if text.lower() in english_words:
            for coordinates, text, confidence in merged_data:
                # Convert coordinates to a suitable format for cv2.rectangle
                x1, y1 = map(int, coordinates[0])
                x2, y2 = map(int, coordinates[2])

                # Draw the rectangle
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Optionally, put the text near the bounding box
                cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                classification = classify_input(text)

                if classification == "Statement" or classification == "ynQuestion" or classification == "whQuestion" or classification == "Reject" or classification == "Emphasis":
                    if text not in seen:
                        print(text)
                        seen.append(text)
                        message = text
                        if message: 
                            messages.append( 
                                {"role": "user", "content": message}, 
                            ) 
                            chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages) 
                        reply = chat.choices[0].message.content 
                        print(f"ChatGPT: {reply}") 
                        messages.append({"role": "assistant", "content": reply}) 
    return image

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform OCR and draw bounding boxes
    processed_frame = perform_ocr_and_draw_boxes(frame)

    # Display the resulting frame
    cv2.imshow('OCR using Webcam', processed_frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
