import os, cv2

BASE = os.path.dirname(__file__)
SRC  = os.path.join(BASE, 'images.jpg')          
OUT1 = os.path.join(BASE, 'mascara.png')
OUT2 = os.path.join(BASE, 'segmentado.png')

img_gray = cv2.imread(SRC, cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError(SRC)

blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

binary = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,      
            11, 2)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)

img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
objetos = 0
for c in contours:
    if cv2.contourArea(c) > 200:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        objetos += 1

print('Objetos detectados:', objetos)

cv2.imwrite(OUT1, opening)
cv2.imwrite(OUT2, img_rgb)
print('Guardados:', OUT1, 'y', OUT2)