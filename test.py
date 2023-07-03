from torchvision import transforms
import torch
import cv2
from os import listdir
from os.path import isfile, join

from index_to_letter import index_to_letter

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
PATH_WEIGHTS = "./weights/best_weights_25.pt"
PATH_TEST = "./test"

asl_model = torch.load(PATH_WEIGHTS)
asl_model.to(DEVICE)
asl_model.eval()

min_w = 20
min_h = 20

# coincidencias
i = 0
j = 0

# Cargar archivo clasificador
hand_cascade = cv2.CascadeClassifier("./Hand_haar_cascade.xml")

onlyfiles = [join(PATH_TEST, f) for f in listdir(PATH_TEST) if isfile(join(PATH_TEST, f))]

transformations=transforms.Compose([
                      transforms.ToPILImage(),
                      transforms.Resize(224),
                      transforms.ToTensor(),
                      transforms.Normalize(
                        mean=[0.485,0.456,0.406], 
                        std=[0.229,0.224,0.225]
                        )
                    ])

for f in onlyfiles:
    img = cv2.imread(f)

    # cambio de color
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # deteccion de la mano
    hands = hand_cascade.detectMultiScale(gray, 1.1, 5)

    for hand in hands[-1:] :
        x, y, w, h = hand 
        if w >= min_w or h >= min_h:
            offset = 10
            hand_section = img[y - offset : y + h + offset, x - offset : x + w + offset]
            
            hand_section_tensor = transformations(hand_section)
            hand_section_tensor = torch.unsqueeze(hand_section_tensor, 0)
            hand_section_tensor = hand_section_tensor.to(DEVICE)
            #print(hand_section_tensor)
            prediction = asl_model(hand_section_tensor)
            index = torch.argmax(prediction).item()
            j += 1
            if index_to_letter[index] in f:
                i += 1
                print(f, " ", index_to_letter[index])
                
print(" Total de coincidencias : ", i, " de ", j)
