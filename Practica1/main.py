from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path = 'dat/fashion-mnist_train.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)
summary = df.head()
print(summary)

# Separate the labels (first column) and image pixel columns
labels = df.iloc[:, 0]  # Assuming the labels are in the first column
image_data = df.iloc[:, 1:]  # Assuming the rest of the columns are image pixels

# Define the dimensions of the images
image_width = 28
image_height = 28

# Create a figure to display the images in a row
plt.figure(figsize=(15, 5))  # Adjust the figure size as needed

# Loop through the first 10 rows and display each image next to each other
for i in range(10):
    plt.subplot(2, 5, i + 1)  # Create a subplot for each image
    image_data_array = np.array(image_data.iloc[i])
    image = image_data_array.reshape((image_height, image_width))
    label = labels.iloc[i]
    plt.imshow(image, cmap='gray')
    plt.title(f'Label: {label}')

plt.show()

X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=2,
                           random_state=5)

# Tractament de les dades: Separació i estandaritzat
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenament i predicció, aquí no hi ha ajustament de paràmetres
clf = SGDClassifier(loss="perceptron", eta0=1, max_iter=1000, learning_rate="constant", random_state=5)
clf.fit(X_train_scaled, y_train)
prediction = clf.predict(X_test_scaled)

# Avaluacio
cf_matrix = confusion_matrix(y_test, prediction)
print(cf_matrix)
accuracy = accuracy_score(y_test, prediction)
print("Accuracy: ", accuracy)