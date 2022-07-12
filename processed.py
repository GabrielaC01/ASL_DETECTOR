import cv2
import os
import numpy as np

DATASET_DIR = "./processed"
DATA = "./data"
min_w = 20
min_h = 20

# Cargar archivo clasificador
hand_cascade = cv2.CascadeClassifier("./Hand_haar_cascade.xml")

# obtener una lista de las subcarpetas en data [A, B, C....]
subfolders = os.listdir(DATA)

# creamos las subcarpetas halladas, en la carpeta processed
for sf in subfolders:
    if not os.path.exists(f"{DATASET_DIR}/{sf}"):
        os.mkdir(f"{DATASET_DIR}/{sf}")

for filename_sf in subfolders:
    # ruta de cada subcarpeta de data
    sf_path = os.path.join(DATA,filename_sf) 
    #nombre de cada imagen de cada subcarpeta
    filename_image = os.listdir(sf_path)

    for file_name in filename_image:
        #ruta de cada imagen de cada subcarpeta de data
        image_path_src = sf_path + "/" + file_name
        image_path_src = os.path.abspath(image_path_src)
        image_path_processed = image_path_src.replace("data", "processed" )
        #leer cada imagen
        img = cv2.imread(image_path_src)
        # cambio de color
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # deteccion de la mano
        hands = hand_cascade.detectMultiScale(gray, 1.3, 5)
        if len(hands) == 0 :
            continue
        hands = sorted(hands, key = lambda f:f[2]*f[3])
        #print (hand)
        # marcado
        for hand in hands[-1:] :
            x, y, w, h = hand 
            if w >= min_w or h >= min_h:
                offset = 10
                hand_section = img[y - offset : y + h + offset, x - offset : x + w + offset]
                hand_section = np.array(hand_section)
                if 0 in hand_section.shape:
                    continue
      
                #print(hand_section)
            
                hand_section = cv2.resize(hand_section, (160, 160))
                print (image_path_processed)
                cv2.imwrite(
                        image_path_processed, hand_section
                )
