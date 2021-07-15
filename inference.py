####### some common libraries #######
import os
import cv2
import numpy as np
from PIL import Image as I

####### torch libraries #######
import torch
import torch.nn.functional as F
from torchvision import transforms

####### models and performance measures #######
from py_files.performance_measure import *
from py_files.RHINet import *
from py_files.ASTMNet import *
from py_files.DSTMNet import *
from py_files.ATTENNet import *
from py_files.ResidualNet import *

TAG = '[inference.py]'
base_dir = os.getcwd()
trained_models_path = os.path.join(base_dir, 'trained_models')
device = 'cpu'

image_sizes = {
    'RHINet': 120,
    'ASTMNet': 120,
    'DSTMNet': 120,
    'ATTENNet': 120,
    'ResidualNet': 224,
}

cache_models = {}

def inference(image, classifier_name):
    size = image_sizes[classifier_name]
    transform_data = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform_data(image)
    image = torch.unsqueeze(image, dim=0)

    if classifier_name not in cache_models:
        model = eval(classifier_name)().to(device)
        checkpoint = torch.load(os.path.join(trained_models_path, f'{classifier_name}.ckpt'), map_location=device)
        model.load_state_dict(checkpoint)
        cache_models[classifier_name] = model
    else:
        model = cache_models[classifier_name]

    class_label = {0: 'non-mitosis', 1: 'mitosis'}

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        pred_outputs = model(image)
        score = pred_outputs.data.cpu().numpy().tolist()[0][1]
        prob = F.softmax(pred_outputs, dim=1)
        score, predicted_label = torch.max(pred_outputs, 1)

        pred_outputs = pred_outputs.data.cpu().numpy().tolist()
        predicted_label = predicted_label.data.cpu().numpy().tolist()
        prob = prob.data.cpu().numpy().tolist()[0]

    return prob[predicted_label[0]], class_label[predicted_label[0]]

if __name__ == '__main__':
    path = './img_dataset/positive_mitosis'
    positives = os.listdir(path)
    for name in positives:
        image_path = os.path.join(path, name)
        image = cv2.imread(image_path)
        output = inference(image, 'ASTMNet')
        print('[output]\n', output)
