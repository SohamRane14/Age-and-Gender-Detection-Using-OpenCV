from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2 as cv
import numpy as np
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Function to detect faces in an image
def get_face_boxes(net, frame, confidence_threshold=0.7):
    """
    Detects faces in a given frame using a deep learning face detector.
    
    Parameters:
    - net: The OpenCV deep learning network for face detection.
    - frame: The image frame in which to detect faces.
    - confidence_threshold: The threshold for confidence to consider a detection valid.
    
    Returns:
    - A tuple with the frame copy and a list of bounding boxes for detected faces.
    """
    frame_copy = frame.copy()  # Create a copy of the frame to avoid modifying the original
    frame_height, frame_width = frame_copy.shape[:2]  # Get dimensions of the frame
    blob = cv.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)
    
    net.setInput(blob)
    detections = net.forward()  # Perform face detection
    boxes = []

    # Loop through each detection and filter by confidence threshold
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            boxes.append([x1, y1, x2, y2])
    
    return frame_copy, boxes

# Function to process an image and predict age and gender
def process_image(image):
    """
    Processes the uploaded image to predict age and gender.
    
    Parameters:
    - image: The image file content to process.
    
    Returns:
    - A list of dictionaries with detected age, gender, and confidence for each face.
    """
    try:
        # Load pre-trained models for age and gender prediction
        age_net = cv.dnn.readNet('models/age_net.caffemodel', 'models/age_deploy.prototxt')
        gender_net = cv.dnn.readNet('models/gender_net.caffemodel', 'models/gender_deploy.prototxt')
        face_net = cv.dnn.readNet('models/opencv_face_detector_uint8.pb', 'models/opencv_face_detector.pbtxt')
    except cv.error as e:
        print(f"Error loading models: {e}")
        return None

    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)  # Mean values for normalization
    age_list = [
        '0-4', '5-9', '10-14', '15-19', '20-24',
        '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74',
        '75-79', '80-84', '85-89', '90-94', '95-99',
        '100+'
    ]
    gender_list = ['Male', 'Female']

    np_image = np.frombuffer(image, np.uint8)  # Convert image bytes to a numpy array
    frame = cv.imdecode(np_image, cv.IMREAD_COLOR)  # Decode the image array to a frame
    if frame is None:
        print(f"Error loading image from buffer")
        return None

    # Detect faces in the image
    _, bboxes = get_face_boxes(face_net, frame)
    results = []
    if not bboxes:
        print("No faces detected in the image.")
    else:
        # For each detected face, predict age and gender
        for bbox in bboxes:
            face = frame[max(0, bbox[1] - 20):min(bbox[3] + 20, frame.shape[0] - 1),
                         max(0, bbox[0] - 20):min(bbox[2] + 20, frame.shape[1] - 1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), model_mean_values, swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()  # Predict gender
            gender = gender_list[gender_preds[0].argmax()]
            gender_confidence = gender_preds[0].max()

            age_net.setInput(blob)
            age_preds = age_net.forward()  # Predict age
            age = age_list[age_preds[0].argmax()]
            age_confidence = age_preds[0].max()

            # Append results for the detected face
            results.append({
                'gender': gender,
                'gender_confidence': float(gender_confidence),
                'age': age,
                'age_confidence': float(age_confidence),
                'bbox': bbox
            })

    return results

# Route to render the main page
@app.route('/')
def index():
    """
    Renders the main page of the web application.
    """
    return render_template('index.html')

# Route to handle image uploads and processing
@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Handles image file uploads, processes the image, and returns results.
    """
    if 'file' not in request.files:
        return redirect(request.url)  # Redirect if no file is provided
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)  # Redirect if file has no name
    if file:
        file_content = file.read()  # Read the file content
        results = process_image(file_content)  # Process the image
        if results:
            encoded_image = base64.b64encode(file_content).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{encoded_image}"
            return jsonify({
                'image_url': image_url,
                'results': results
            })
        else:
            return jsonify({'error': 'Failed to process image'})
    return redirect(url_for('index'))

# Route to handle camera frame processing
@app.route('/camera', methods=['POST'])
def process_camera():
    """
    Handles camera frame data, processes the frame, and returns results.
    """
    data = request.get_json()
    if 'frame' not in data:
        return jsonify({'error': 'No frame data'})
    
    frame_data = base64.b64decode(data['frame'])  # Decode base64 frame data
    results = process_image(frame_data)  # Process the frame
    if results:
        return jsonify({
            'results': results
        })
    else:
        return jsonify({'error': 'Failed to process frame'})

# Route to check the health of the server
@app.route('/health', methods=['GET'])
def health():
    """
    Returns the server health status.
    """
    return "OK", 200

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask application with debug mode enabled
