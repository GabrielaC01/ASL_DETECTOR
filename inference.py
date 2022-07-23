from cv2 import normalize
from numpy import imag
from dataset import PATH_VALID
from PIL import Image
from index_to_letter import index_to_letter
from torchvision import transforms

import torch
import cv2
import numpy as np


DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
PATH_WEIGHTS = "./weights/best_weights_15.pt"

'''
asl_model = ASL_Model(n_classes = len (trainset.class_to_idx)).to(DEVICE) 
asl_model.load_state_dict(torch.load(PATH_WEIGHTS))
'''
asl_model = torch.load(PATH_WEIGHTS)
asl_model.to(DEVICE)
asl_model.eval()

min_w = 20
min_h = 20

# Cargar archivo clasificador
hand_cascade = cv2.CascadeClassifier("./Hand_haar_cascade.xml")

img = cv2.imread("./test/A_10.jpg")

# cambio de color
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# deteccion de la mano
hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

for hand in hands[-1:] :
    x, y, w, h = hand 
    if w >= min_w or h >= min_h:
        offset = 10
        hand_section = img[y - offset : y + h + offset, x - offset : x + w + offset]
        '''''
        hand_section = np.array(hand_section)
        if 0 in hand_section.shape:
            continue
    '''''
    
        hand_section = cv2.resize(hand_section, (224, 224))
        #hand_section = np.array(hand_section)
        #print(hand_section.shape)
        #hand_section_color = cv2.cvtColor(hand_section, cv2.COLOR_GRAY2BGR)
        #ancho x alto x canales
        #canales x ancho x alto
        hand_section_tensor = torch.from_numpy(hand_section).permute(2, 0, 1) 
        #batch x canales x ancho x alto
        #normalize =  transforms.Compose([    
        #              transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        #])
        #print(hand_section_tensor)
        #mean, std = np.array([0.485,0.456,0.406]), np.array([0.229,0.224,0.225])
        #hand_section_tensor = (hand_section_tensor - mean) / std
        #print (hand_section/255.)
        hand_section_tensor = torch.unsqueeze(hand_section_tensor/255., 0).float()
       # hand_section_tensor = torch.unsqueeze(hand_section_tensor, 0).float()
        hand_section_tensor = hand_section_tensor.to(DEVICE)
        prediction = asl_model(hand_section_tensor)
        index = torch.argmax(prediction).item()
        print ( index) 
        print(index_to_letter[index])

