import ssl
import certifi
import urllib.request
import easyocr

# Use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)

import cv2
import numpy as np
import threading
import difflib

# Predefined list of common English words
common_english_words = [
    "hello", "world", "example", "text", "detected", "word", "guess",
    "ocr", "python", "code", "camera", "video", "frame", "process",
    "language", "model", "simple", "common", "english", "words", "list", "hi", "amazing", "enthalpy"
]

reader = easyocr.Reader(['en'])

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary

def ocr_processing(frame, result):
    processed_frame = preprocess_frame(frame)
    detections = reader.readtext(processed_frame, paragraph=True)
    result.extend(detections)

def get_closest_word(word, word_list):
    matches = difflib.get_close_matches(word, word_list, n=1, cutoff=0.7)
    if matches:
        return matches[0]
    return None

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream from webcam.")
        return

    frame_interval = 5  # Process every 5th frame to reduce lag
    frame_count = 0
    ocr_results = []

    last_detected_words = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame = cv2.resize(frame, (640, 480))
        x, y, w, h = 100, 100, 440, 280
        roi = frame[y:y+h, x:x+w]

        if frame_count % frame_interval == 0:
            ocr_thread = threading.Thread(target=ocr_processing, args=(roi, ocr_results))
            ocr_thread.start()
            ocr_thread.join()  # Ensure the thread has completed

        if ocr_results:
            latest_detections = ocr_results[-len(ocr_results):]

            current_detected_words = set()

            for detection in latest_detections:
                if len(detection) == 2:
                    bbox, text = detection
                else:
                    bbox, text, _ = detection

                text = text.strip()
                if text:
                    # Guess the words and filter them
                    words_in_text = text.split()
                    guessed_words = [get_closest_word(word, common_english_words) for word in words_in_text]
                    guessed_words = [word for word in guessed_words if word is not None]
                    guessed_text = ' '.join(guessed_words)

                    # Only print if there are guessed words
                    if guessed_words:
                        for word in guessed_words:
                            if word not in last_detected_words:
                                last_detected_words.add(word)
                                current_detected_words.add(word)

                        # Convert to lowercase and check for "hello"
                        if "hello" in guessed_text.lower():
                            print("Detected 'hello', exiting...")
                            cap.release()
                            cv2.destroyAllWindows()
                            return
                        
            if current_detected_words:
                print("Last detected words:", ' '.join(current_detected_words))
                last_detected_words = current_detected_words

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
