import cv2 as cv
import argparse
import tkinter as tk
from tkinter import filedialog
import time

def get_face_boxes(net, frame, confidence_threshold=0.7):
    frame_copy = frame.copy()
    frame_height, frame_width = frame_copy.shape[:2]
    blob = cv.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections = net.forward()
    boxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            boxes.append([x1, y1, x2, y2])
            cv.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame_copy, boxes

def process_image(image_path):
    # Load the pre-trained models
    try:
        age_net = cv.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
        gender_net = cv.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')
        face_net = cv.dnn.readNet('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
    except cv.error as e:
        print(f"Error loading models: {e}")
        return

    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = [
        '0-4', '5-9', '10-14', '15-19', '20-24',
        '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74',
        '75-79', '80-84', '85-89', '90-94', '95-99',
        '100+'
    ]
    gender_list = ['Male', 'Female']

    # Load the image
    frame = cv.imread(image_path)
    if frame is None:
        print(f"Error loading image: {image_path}")
        return

    frame_with_faces, bboxes = get_face_boxes(face_net, frame)
    if not bboxes:
        print("No faces detected in the image.")
    else:
        for bbox in bboxes:
            face = frame[max(0, bbox[1] - 20):min(bbox[3] + 20, frame.shape[0] - 1),
                         max(0, bbox[0] - 20):min(bbox[2] + 20, frame.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print(f"Gender: {gender}, Confidence: {gender_preds[0].max():.3f}")

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print(f"Age: {age}, Confidence: {age_preds[0].max():.3f}")

            label = f"{gender}, {age}"
            cv.putText(frame_with_faces, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow("Age Gender Detection", frame_with_faces)
        cv.waitKey(0)
        cv.destroyAllWindows()

def process_camera():
    # Load the pre-trained models
    try:
        age_net = cv.dnn.readNet('age_net.caffemodel', 'age_deploy.prototxt')
        gender_net = cv.dnn.readNet('gender_net.caffemodel', 'gender_deploy.prototxt')
        face_net = cv.dnn.readNet('opencv_face_detector_uint8.pb', 'opencv_face_detector.pbtxt')
    except cv.error as e:
        print(f"Error loading models: {e}")
        return

    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = [
        '0-4', '5-9', '10-14', '15-19', '20-24',
        '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74',
        '75-79', '80-84', '85-89', '90-94', '95-99',
        '100+'
    ]
    gender_list = ['Male', 'Female']

    # Open camera capture
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    padding = 20
    detected_once = False  # Track whether age and gender have been detected and shown
    while not detected_once:
        start_time = time.time()
        has_frame, frame = cap.read()
        if not has_frame:
            print("No frame captured. Exiting...")
            break

        frame_with_faces, bboxes = get_face_boxes(face_net, frame)
        if not bboxes:
            print("No faces detected. Moving to the next frame...")
            continue

        for bbox in bboxes:
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                         max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = gender_list[gender_preds[0].argmax()]
            print(f"Gender: {gender}, Confidence: {gender_preds[0].max():.3f}")

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            print(f"Age: {age}, Confidence: {age_preds[0].max():.3f}")

            label = f"{gender}, {age}"
            cv.putText(frame_with_faces, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
            detected_once = True  # Mark that the information has been detected and shown
            break  # Exit the loop once we have detected and processed a face

        cv.imshow("Age Gender Detection", frame_with_faces)
        print(f"Processing time: {time.time() - start_time:.3f}s")
        cv.waitKey(0)  # Wait for any key to exit after showing the result

    cap.release()
    cv.destroyAllWindows()

def main():
    # Initialize Tkinter root window and hide it
    root = tk.Tk()
    root.withdraw()

    print("Choose input source:")
    print("1. Upload an image file")
    print("2. Use the camera")

    choice = input("Enter your choice (1 or 2): ")
    if choice == '1':
        print("Please select an image file.")
        image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            process_image(image_path)
        else:
            print("No image selected.")
    elif choice == '2':
        process_camera()
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()