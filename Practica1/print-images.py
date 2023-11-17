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