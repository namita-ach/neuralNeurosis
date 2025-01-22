import os
import tensorflow as tf
from tensorflow.keras import layers, models, applications
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2

# Clone the dataset repository
!git clone https://github.com/SartajBhuvaji/Brain-Tumor-Classification-DataSet.git

data_dir = "/home/pes1ug22am100/Documents/Research and Experimentation/neuralNeurosis/Brain-Tumor-Classification-DataSet"
img_size = (224, 224)
batch_size = 32

# Data preparation function
def prepare_data(train_dir, test_dir, img_size, batch_size):
    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, test_gen

train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")
train_gen, test_gen = prepare_data(train_dir, test_dir, img_size, batch_size)
num_classes = len(train_gen.class_indices)

# Print dataset information
print(f"Number of classes: {len(train_gen.class_indices)}")
print(f"Class labels: {train_gen.class_indices}")

# Encoding visualization
def visualize_encoding(image_path, encoding_type='rate', time_window=100, max_spikes=20):
    if encoding_type == 'rate':
        encode_function = encode_rate_coding
    elif encoding_type == 'temporal':
        encode_function = encode_temporal_coding
    else:
        raise ValueError(f"Unsupported encoding type: {encoding_type}")

    spike_train = encode_function(image_path, time_window, max_spikes)
    img = image.load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = np.array(img) / 255.0

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_array, cmap='gray')
    plt.title('Original MRI Image')

    plt.subplot(1, 2, 2)
    if encoding_type == 'rate':
        plt.imshow(np.sum(spike_train, axis=2), cmap='hot', interpolation='nearest')
        plt.title('Rate Coding Encoded Image')
    elif encoding_type == 'temporal':
        plt.imshow(np.argmax(spike_train, axis=2), cmap='hot', interpolation='nearest')
        plt.title('Temporal Coding Encoded Image')

    plt.colorbar()
    plt.show()

# Rate encoding
def encode_rate_coding(image_path, time_window=100, max_spikes=20):
    img = image.load_img(image_path, target_size=(128, 128), color_mode='grayscale')
    img_array = np.array(img) / 255.0

    spike_train = np.zeros((img_array.shape[0], img_array.shape[1], time_window))

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            num_spikes = int(img_array[i, j] * max_spikes)
            spike_train[i, j, :num_spikes] = 1

    return spike_train

# Random class sample visualization
def random_class_sample(train_gen, encoding_type='rate', num_samples=1, time_window=100, max_spikes=20):
    class_names = list(train_gen.class_indices.keys())

    for class_name in class_names:
        class_dir = os.path.join(train_dir, class_name)
        images = os.listdir(class_dir)
        random_image = random.choice(images)
        image_path = os.path.join(class_dir, random_image)

        print(f"Displaying sample from class: {class_name}")
        visualize_encoding(image_path, encoding_type, time_window, max_spikes)

# Additional setup and functionality can be added here.
