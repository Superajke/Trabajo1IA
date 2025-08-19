# -*- coding: utf-8 -*-
"""
Reto 5 – Filtrado de ruido sal-pimienta
A) Filtros “a mano”            B) Comparación con cv2.medianBlur
"""

import os, random, math
import cv2, matplotlib.pyplot as plt, numpy as np

# 1)  Cargar imagen -----------------------------------------------------------
BASE = os.path.dirname(__file__)          # .../Trabajo 1/Reto5
FILE = os.path.join(BASE, 'images.jpg')   # cámbialo si usas otro nombre
img  = cv2.imread(FILE, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(FILE)

# 2)  Añadir ruido sal-pimienta -----------------------------------------------
def salt_pepper(src, prob=0.02):
    out = src.copy()
    h, w = out.shape
    n = int(prob * h * w)
    for _ in range(n // 2):
        y, x = random.randrange(h), random.randrange(w); out[y, x] = 255
    for _ in range(n // 2):
        y, x = random.randrange(h), random.randrange(w); out[y, x] = 0
    return out

noisy = salt_pepper(img, 0.02)

# 3)  Utilidades para convoluciones manuales ----------------------------------
def pad(a, k):  p = k // 2;  return cv2.copyMakeBorder(a, p,p,p,p, cv2.BORDER_REPLICATE)

def convolve(src, kernel):
    k = len(kernel);  src_p = pad(src, k);  dst = src.copy()
    h, w = src.shape
    for y in range(h):
        for x in range(w):
            acc = 0
            for i in range(k):
                for j in range(k):
                    acc += kernel[i][j] * src_p[y+i, x+j]
            dst[y, x] = int(acc)
    return dst

# 3A) Media 3×3
kernel_mean  = [[1/9]*3]*3
mean         = convolve(noisy, kernel_mean)

# 3B) Gauss 3×3
kernel_gauss = [[1,2,1],[2,4,2],[1,2,1]]
kernel_gauss = [[v/16 for v in row] for row in kernel_gauss]
gauss        = convolve(noisy, kernel_gauss)

# 3C) Mediana manual 3×3
def median_filter(src, k=3):
    p = k//2; src_p = pad(src, k); dst = src.copy()
    h, w = src.shape
    for y in range(h):
        for x in range(w):
            block = [src_p[y+i, x+j] for i in range(k) for j in range(k)]
            block.sort();  dst[y, x] = block[len(block)//2]
    return dst

median_manual = median_filter(noisy, 3)

# 4)  Mediana optimizada OpenCV ----------------------------------------------
median_cv = cv2.medianBlur(noisy, 3)

# 5)  PSNR --------------------------------------------------------------------
def psnr(ref, test):
    mse = np.mean((ref.astype('float32') - test.astype('float32')) ** 2)
    if mse == 0:  return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

print('\nPSNR respecto a la imagen limpia (dB):')
for nombre, img_f in [('Media', mean),
                      ('Gauss', gauss),
                      ('Mediana manual', median_manual),
                      ('Mediana OpenCV', median_cv)]:
    print('{:>15}: {:.2f}'.format(nombre, float(psnr(img, img_f))))

# 6)  Guardar evidencias ------------------------------------------------------
cv2.imwrite(os.path.join(BASE, 'noisy.png'),        noisy)
cv2.imwrite(os.path.join(BASE, 'mean.png'),         mean)
cv2.imwrite(os.path.join(BASE, 'gauss.png'),        gauss)
cv2.imwrite(os.path.join(BASE, 'med_manual.png'),   median_manual)
cv2.imwrite(os.path.join(BASE, 'med_cv.png'),       median_cv)

# 7)  Figura comparativa ------------------------------------------------------
titles  = ['Original', 'Ruido S&P', 'Media', 'Gauss',
           'Mediana manual', 'Mediana OpenCV']
images  = [img, noisy, mean, gauss, median_manual, median_cv]

plt.figure(figsize=(12,8))
for i, (t, im) in enumerate(zip(titles, images)):
    plt.subplot(2, 3, i+1); plt.imshow(im, cmap='gray')
    plt.title(t); plt.axis('off')
plt.tight_layout(); plt.show()