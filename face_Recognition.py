
import keras
import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.src.legacy.preprocessing.image import ImageDataGenerator


# Get data from the webcam
cascade_path = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

def get_data():
    count = 0  # Counter variable to name the image files
    cam = cv2.VideoCapture(0)  # Initialize the camera to capture video from the default camera (camera 0)
    while True:  # Infinite loop to continuously read data from the camera
        ok, frame = cam.read()  # Read a frame from the camera
        faces = cascade_path.detectMultiScale(frame, 1.3, 5)  # Detect faces in the frame
        
        for (x, y, w, h) in faces:  # Iterate over each detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # Draw a rectangle around the face
            # Crop and resize the face
            img = cv2.resize(frame[y+2:y+h-2, x+2:x+w-2], (128, 128))
            # Save the face image to the directory 'datasets/train_data/Tuan_Anh' with the name count.jpg
            cv2.imwrite(f'datasets/train_data/Tuan_Anh/{count}.jpg', img)
            cv2.imshow('FRAME', frame)  # Display the frame with the rectangle around the face
            count += 1  # Increment the counter for the next file name
        
        if cv2.waitKey(1) & 0xFF == ord('v'):  # If the 'v' key is pressed, break the loop
            break
    
    cam.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV display windows


# Create the model training function
def get_OHC(label, cnt, index):
    tmp = [0] * cnt  # Initialize a list of zeros with length 'cnt'
    tmp[index-1] = 1  # Set the element at position 'index-1' to 1
    label_OHC = [label, tmp]  # Create a list containing the label and the one-hot encoded list
    return label_OHC  


import os
import cv2

def data_processing():
    DATA_PATH = 'datasets/train_data'  # Define the path to the dataset
    count, index = 0, 0  # Initialize count and index variables
    data, list_label = [], []  # Initialize lists to store data and labels

    # First loop: Count the number of persons (subdirectories) in the dataset
    for person in os.listdir(DATA_PATH):
        person_path = os.path.join(DATA_PATH, person)
        count += 1

    # Second loop: Process each person's images and create one-hot encoded labels
    for person in os.listdir(DATA_PATH):
        index += 1
        person_path = os.path.join(DATA_PATH, person)
        label = person  # Use the person's name as the label
        label_OHC = get_OHC(label, count, index)  # Create one-hot encoded label
        list_label.append(label_OHC)  # Append the one-hot encoded label to the list

        # Process each image file for the current person
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            img = cv2.imread(image_path)  # Read the image using OpenCV
            data.append((img, label_OHC[1]))  # Append the image and its one-hot encoded label to the data list

    return data, list_label  # Return the processed data and list of one-hot encoded labels



# Build the model
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.15),

        layers.Conv2D(512, (3, 3), activation='relu'),
        layers.MaxPool2D((2, 2)),
        layers.Dropout(0.1),

        layers.Flatten(),
        layers.Dense(1000, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Process data
data, list_label = data_processing()
np.random.shuffle(data)
data_train, data_test = train_test_split(data, test_size=0.3, random_state=42)
x_train = np.array([x[0] for x in data_train]).astype(np.float32) / 255.0
y_train = np.array([x[1] for x in data_train]).astype(np.float32)
x_test = np.array([x[0] for x in data_test]).astype(np.float32) / 255.0
y_test = np.array([x[1] for x in data_test]).astype(np.float32)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Train the model
model = build_model()
model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=10, validation_data=(x_test, y_test))
model.save('Face_Recognition_10epochs.keras')

models_10p = models.load_model('Face_Recognition_10epochs.keras')

# Predict images by camera
cam = cv2.VideoCapture(0)
ok, frame = cam.read()
if not ok:
    print("Failed to capture image")
    cam.release()
    cv2.destroyAllWindows()
    exit()

# Perform face detection
faces = cascade_path.detectMultiScale(frame, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    img = cv2.resize(frame[y+2:y+h-2, x+2:x+w-2], (128, 128))  # Corrected slicing for proper dimensions
    img = img / 255.0  # Normalize image if your model expects normalized inputs
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match input shape
    result = np.argmax(models_10p.predict(img))
    cv2.putText(frame, list_label[result][0], (x+35, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 2)

# Display the frame with predictions
cv2.imshow('FRAME', frame)
cv2.waitKey(0)  # Wait for a key press to close the window

# Release the webcam and close all OpenCV windows
cam.release()
cv2.destroyAllWindows()


