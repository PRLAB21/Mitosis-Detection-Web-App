import os
import time
import base64
from PIL import Image
from flask import Flask, request
from flask_cors import CORS

import inference

def read_image(image_data):
    TAG2 = TAG + '[read_image]'
    print(TAG2, '[starts]')
    image_data = base64.decodebytes(image_data)
    with open('temp_image.jpg', 'wb') as f:
        f.write(image_data)
        f.close()
    image = Image.open('temp_image.jpg').convert('RGB')
    os.remove('temp_image.jpg')
    return image

app = Flask(__name__)
CORS(app)

TAG = f'[{__name__}]'
BASE_WORK_DIR = os.getcwd()

# This is the server function to handle requests and get images from client
@app.route('/inference', methods=['POST'])
def inference_handle():
    TAG2 = TAG + '[inference]'
    print(TAG2, '[starts]')
    
    # check for valid request
    if not request.json:
        return 'Server Error!', 500

    # STEP 1: process input image
    header_len = len('data:image/jpeg;base64,')
    image_data = request.json['image_data'][header_len:].encode()
    classifier_name = request.json['classifier_name']
    t1 = time.time()
    image = read_image(image_data)
    print(TAG2, '[read_image][time_taken]', time.time() - t1)

    # STEP 2: perform detection
    t1 = time.time()
    score, class_label = inference.inference(image, classifier_name)
    print(TAG2, '[inference][time_taken]', time.time() - t1)

    prediction = {'score': score, 'class_label': class_label, 'status': 'Operation Completed without Error'}
    print(TAG, '[prediction]\n', prediction)
    return prediction, 200

@app.route('/hello')
def index():
    return 'Welcome'

if __name__ == '__main__':
    app.run(debug=True)
