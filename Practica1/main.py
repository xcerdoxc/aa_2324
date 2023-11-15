import os
from sklearn.calibration import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray
import numpy as np

# Step 1: Load and Preprocess the Image Data
def load_and_preprocess(folder_path, target_size=(64, 64)):
    images = []
    labels = []
    
    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = io.imread(image_path)
            
            # Preprocess the image (resize, convert to grayscale, etc.)
            image = resize(image, target_size)
            # Check if the image is already grayscale
            if image.ndim == 2:
                # Grayscale image
                images.append(image.flatten())
            else:
                # Convert color image to grayscale
                gray_image = rgb2gray(image)
                images.append(gray_image.flatten())

            labels.append(label)
    
    return np.array(images), np.array(labels)

# Step 2: Load and Label the Data
train_path = 'dat/train/'
test_path = 'dat/test/'
X_train, y_train = load_and_preprocess(train_path)
print("Imatges train processades")
X_test, y_test = load_and_preprocess(test_path)
print("Imatges test processades")

# Step 3: Scale the Data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
print("Imatges escalades")

# Step 4: Train the SVC Model
kernels = ['linear', 'rbf', 'poly']

for kernel in kernels:
    print(f"TEST KERNEL: {kernel}")
    svm = SVC(kernel=kernel)
    svm.fit(X_train_scaled, y_train)
    y_pred = svm.predict(X_test_scaled)

    # Step 5: Evaluate the Model
    cf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion matrix:\n", cf_matrix)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy: ", accuracy)