import random
WIDTH = HEIGHT = 1000

def crear_matriz(w=WIDTH, h=HEIGHT):
    return [[random.randint(0, 255) for _ in range(w)] for _ in range(h)]

mat = crear_matriz()

def estadisticos(m):
    total = 0
    n = len(m) * len(m[0])
    vmin = 255
    vmax = 0

    for fila in m:
        for val in fila:
            if val < vmin: vmin = val
            if val > vmax: vmax = val
            total += val
    media = total / n

    suma_sq = 0
    for fila in m:
        for val in fila:
            diff = val - media
            suma_sq += diff * diff
    varianza = suma_sq / n
    desvest = varianza ** 0.5
    return vmin, vmax, media, desvest

vmin, vmax, media, desvest = estadisticos(mat)
print(vmin, vmax, round(media,2), round(desvest,2))

from PIL import Image
import matplotlib.pyplot as plt

img = Image.frombytes('L', (WIDTH, HEIGHT), bytes(sum(mat, [])))
plt.imshow(img, cmap='gray')
plt.title('Matriz aleatoria 1000×1000')
plt.axis('off')
plt.show()
img.save('matriz_aleatoria.png')

from google.colab import drive
drive.mount('Novia_Hermosa.jpg')
from PIL import Image
img = Image.open('/content/drive/MyDrive/matriz_aleatoria.png')
plt.imshow(img, cmap='gray'); plt.axis('off')
