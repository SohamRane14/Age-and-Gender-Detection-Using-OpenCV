from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2 as cv
import numpy as np
import base64
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

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
    
    return frame_copy, boxes

def process_image(image):
    try:
        # Update paths to model files
        age_net = cv.dnn.readNet('models/age_net.caffemodel', 'models/age_deploy.prototxt')
        gender_net = cv.dnn.readNet('models/gender_net.caffemodel', 'models/gender_deploy.prototxt')
        face_net = cv.dnn.readNet('models/opencv_face_detector_uint8.pb', 'models/opencv_face_detector.pbtxt')
    except cv.error as e:
        print(f"Error loading models: {e}")
        return None

    model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)
    age_list = [
        '0-4', '5-9', '10-14', '15-19', '20-24',
        '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74',
        '75-79', '80-84', '85-89', '90-94', '95-99',
        '100+'
    ]
    gender_list = ['Male', 'Female']

    np_image = np.frombuffer(image, np.uint8)
    frame = cv.imdecode(np_image, cv.IMREAD_COLOR)
    if frame is None:
        print(f"Error loading image from buffer")
        return None

    _, bboxes = get_face_boxes(face_net, frame)
    results = []
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
            gender_confidence = gender_preds[0].max()

            age_net.setInput(blob)
            age_preds = age_net.forward()
            age = age_list[age_preds[0].argmax()]
            age_confidence = age_preds[0].max()

            results.append({
                'gender': gender,
                'gender_confidence': float(gender_confidence),
                'age': age,
                'age_confidence': float(age_confidence),
                'bbox': bbox
            })

    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_content = file.read()
        results = process_image(file_content)
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

@app.route('/camera', methods=['POST'])
def process_camera():
    data = request.get_json()
    if 'frame' not in data:
        return jsonify({'error': 'No frame data'})
    
    frame_data = base64.b64decode(data['frame'])
    results = process_image(frame_data)
    if results:
        return jsonify({
            'results': results
        })
    else:
        return jsonify({'error': 'Failed to process frame'})

if __name__ == '__main__':
    app.run(debug=True)
