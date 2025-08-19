import numpy as np
import cv2

print(cv2.__version__)

img = cv2.imread('Novia_Hermosa.jpg',0)
height, width = img.shape[:2]
cv2.imshow('Foto',img)
print('Altura', height)
print('Ancho', width)
cv2.waitKey(0)
