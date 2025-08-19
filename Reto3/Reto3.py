import glob, os, cv2, numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump

def descriptor_hog(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("No se pudo leer {path}")
    img = cv2.resize(img, (128, 128))          
    hog = cv2.HOGDescriptor()                  
    return hog.compute(img).flatten()         

X, y = [], []
base_dir = os.path.dirname(__file__)          

for cls, label in [('Buenos', 0), ('Malos', 1)]:
    pattern = os.path.join(base_dir, cls, '*.*')  
    for path in glob.glob(pattern):
        X.append(descriptor_hog(path))
        y.append(label)

if len(X) < 4:
    raise RuntimeError("Necesitas ≥2 imágenes por clase para continuar")

print("Total imágenes: {len(X)} | Buenos: {y.count(0)} | Malos: {y.count(1)}")

Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)

clf = LinearSVC(dual=False)
clf.fit(Xtr, ytr)

acc = clf.score(Xte, yte)
print("Accuracy: {acc:.3f}")
    
print("\nMatriz de confusión (filas = real, cols = predicho):")
print(confusion_matrix(yte, clf.predict(Xte)))

print("\nReporte por clase:")
print(classification_report(yte, clf.predict(Xte), target_names=['Buenos', 'Malos']))

dump(clf, os.path.join(base_dir, 'svm_hog.joblib'))
print("Modelo guardado en svm_hog.joblib")