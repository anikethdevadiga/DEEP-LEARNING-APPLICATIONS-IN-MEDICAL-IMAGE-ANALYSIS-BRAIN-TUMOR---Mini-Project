# Import necessary libraries
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import imutils
from keras.utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
import ftplib
from tkinter import ttk

# Initialize the main Tkinter window
main = tkinter.Tk()
main.title("Identifying Brain Tumor using X-Ray Images") # Designing main screen
main.geometry("1300x1200")

# Global variables
global filename
global accuracy
X = []  # To store image data
Y = []  # To store labels
global classifier
disease = ['No Tumor Detected', 'Tumor Detected']  # Class labels

# Ensure Model directory exists
if not os.path.exists('Model'):
    os.makedirs('Model')

# Load the pre-trained segmentation model
segmented_model = None
seg_model_json = os.path.join('Model', 'segmented_model.json')
seg_model_weights = os.path.join('Model', 'segmented.weights.h5')
if os.path.exists(seg_model_json) and os.path.exists(seg_model_weights):
    with open(seg_model_json, "r") as json_file:
        loaded_model_json = json_file.read()
        segmented_model = model_from_json(loaded_model_json)
    segmented_model.load_weights(seg_model_weights)
else:
    print("Segmentation model files not found. Segmentation will be disabled.")

# Function to perform edge detection on the segmented image
def edgeDetection():
    img = cv2.imread('myimg.png', 0)  # Load the segmented image as grayscale
    orig = cv2.imread('test1.png')    # Load the original image
    # Ensure the mask is binary
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    result = orig.copy()
    found = False
    for c in contours:
        area = cv2.contourArea(c)
        print(f"Contour area: {area}")
        if area > 100:  # Draw circle for any reasonably sized contour
            found = True
            (x, y), radius = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(result, center, radius, (0, 0, 255), 8)  # Red circle, thick
            cv2.drawContours(result, [c], -1, (0, 255, 255), 2)  # Highlight contour
    if not found:
        print("No tumor contour found.")
    cv2.imwrite("debug_edge.png", result)  # Save for inspection
    return result   

# Function to segment the tumor from the input image
def tumorSegmentation(filename):
    global segmented_model
    if segmented_model is None:
        text.insert(END, "Segmentation model not loaded!\n")
        return None, None
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, 64, 64, 1)
    img = (img - 127.0) / 127.0
    preds = segmented_model.predict(img)[0]
    # Convert mask to binary
    mask = (preds > 0.5).astype(np.uint8) * 255
    orig = cv2.imread(filename, 0)
    orig = cv2.resize(orig, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("test1.png", orig)
    segmented_image = cv2.resize(mask, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("myimg.png", segmented_image)# Save the segmented image
    edge_detection = edgeDetection()  # Perform edge detection
    return segmented_image, edge_detection  # Return the segmented and edge-detected images

# Function to upload the dataset
def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")  # Open a dialog to select the dataset directory
    text.delete('1.0', END)  # Clear the text box
    text.insert(END, filename + " loaded\n")  # Display the selected directory

# Function to preprocess the dataset and extract features
def datasetPreprocessing():
    global X, Y
    X.clear()
    Y.clear()
    # Use consistent file names for saving and loading
def datasetPreprocessing():
    global X, Y
    X.clear()
    Y.clear()
    data_path = os.path.join('Model', 'myimg_data.txt.npy')
    label_path = os.path.join('Model', 'myimg_label.txt.npy')
    if os.path.exists(data_path) and os.path.exists(label_path):
        X = np.load(data_path)
        Y = np.load(label_path)
    else:
        # Process images from the "no tumor" directory
        no_dir = os.path.join(filename, "no")
        yes_dir = os.path.join(filename, "yes")
        if not os.path.exists(no_dir) or not os.path.exists(yes_dir):
            text.insert(END, "Dataset folders 'no' and 'yes' not found!\n")
            return
        for name in os.listdir(no_dir):
            img_path = os.path.join(no_dir, name)
            img = cv2.imread(img_path, 0)
            if img is None:
                continue
            ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.resize(img, (128, 128))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(128, 128, 1)
            X.append(im2arr)
            Y.append(0)
        for name in os.listdir(yes_dir):
            img_path = os.path.join(yes_dir, name)
            img = cv2.imread(img_path, 0)
            if img is None:
                continue
            ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.resize(img, (128, 128))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(128, 128, 1)
            X.append(im2arr)
            Y.append(1)
        X = np.asarray(X)
        Y = np.asarray(Y)
        np.save("Model/myimg_data.txt.npy", X)
        np.save("Model/myimg_label.txt.npy", Y)
    print(X.shape)
    print(Y.shape)
    print(Y)
    if len(X) > 20:
        cv2.imshow('Sample Image', X[20])
        cv2.waitKey(0)
    text.insert(END, "Total number of images found in dataset: " + str(len(X)) + "\n")
    text.insert(END, "Total number of classes: " + str(len(set(Y))) + "\n\n")
    text.insert(END, "Class labels found in dataset: " + str(disease))
        
# Function to train the CNN model for tumor detection
def trainTumorDetectionModel():
    global accuracy, classifier
    if len(X) == 0 or len(Y) == 0:
        text.insert(END, "Please preprocess the dataset first!\n")
        return
    YY = to_categorical(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    x_train = X[indices]
    y_train = YY[indices]
    model_json_path = os.path.join('Model', 'model.json')
    model_weights_path = os.path.join('Model', 'model.weights.h5')
    history_path = os.path.join('Model', 'history.pckl')
    if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
        with open(model_json_path, "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights(model_weights_path)
    else:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
        classifier = Sequential()
        classifier.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 1), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dense(units=2, activation='softmax'))
        print(classifier.summary())
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        classifier.save_weights(model_weights_path)
        model_json = classifier.to_json()
        with open(model_json_path, "w") as json_file:
            json_file.write(model_json)
        with open(history_path, 'wb') as f:
            pickle.dump(hist.history, f)
    if os.path.exists(history_path):
        with open(history_path, 'rb') as f:
            data = pickle.load(f)
        acc = data['accuracy']
        accuracy = acc[4] * 100
        text.insert(END, '\n\nCNN Brain Tumor Model Generated. See black console to view layers of CNN\n\n')
        text.insert(END, "CNN Brain Tumor Prediction Accuracy on Test Images: " + str(accuracy) + "\n")
    else:
        text.insert(END, "Training history not found!\n")


# Function to classify an input image and detect tumors
def tumorClassification():
    test_images_dir = os.path.join(os.path.dirname(__file__), 'testImages')
    filename_img = filedialog.askopenfilename(initialdir=test_images_dir)
    img = cv2.imread(filename_img, 0)
    if img is None:
        text.insert(END, "Image not found or cannot be opened!\n")
        return
    img = cv2.resize(img, (128, 128))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 128, 128, 1)
    XX = np.asarray(im2arr)
    predicts = classifier.predict(XX)
    print(predicts)
    cls = np.argmax(predicts)
    print(cls)
    img_color = cv2.imread(filename_img)
    img_color = cv2.resize(img_color, (800, 500))
    cv2.putText(img_color, 'Classification Result: ' + disease[cls], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow('Classification Result: ' + disease[cls], img_color)
    if cls == 1 and segmented_model is not None:
        segmented_image, edge_image = tumorSegmentation(filename_img)
        cv2.imshow("Tumor Segmented Image", segmented_image)
        cv2.imshow("Edge Detected Image", edge_image)
    cv2.waitKey(0)

# Function to plot the training accuracy and loss graph
def graph():
    # Get the absolute path to the Model directory
    model_dir = os.path.join(os.path.dirname(__file__), 'Model')
    history_path = os.path.join(model_dir, 'history.pckl')
    print("Looking for training history at:", history_path)
    if not os.path.exists(history_path):
        text.insert(END, "Training history file not found!\n")
        return
    try:
        with open(history_path, 'rb') as f:
            data = pickle.load(f)
        accuracy = data['accuracy']
        loss = data['loss']
        plt.figure(figsize=(10, 6))
        plt.grid(True)
        plt.xlabel('Training Epoch')
        plt.ylabel('Accuracy/Loss')
        plt.plot(loss, 'ro-', color='red')
        plt.plot(accuracy, 'ro-', color='green')
        plt.legend(['Loss', 'Accuracy'], loc='upper left')
        plt.title('Brain Tumor CNN Model Training Accuracy & Loss Graph')
        plt.show()
    except Exception as e:
        text.insert(END, f"Error loading training history: {e}\n")

# GUI Design
font = ('times', 16, 'bold')
title = Label(main, text='Identifying Brain Tumor using X-Ray Images')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = Text(main, height=20, width=150)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

# Buttons for various functionalities
uploadButton = Button(main, text="Upload Tumor X-Ray Images Dataset", command=uploadDataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)  

preprocessButton = Button(main, text="Dataset Preprocessing & Features Extraction", command=datasetPreprocessing)
preprocessButton.place(x=430, y=550)
preprocessButton.config(font=font1) 

cnnButton = Button(main, text="Trained CNN Brain Tumor Detection Model", command=trainTumorDetectionModel)
cnnButton.place(x=810, y=550)
cnnButton.config(font=font1) 

classifyButton = Button(main, text="Brain Tumor Segmentation & Classification", command=tumorClassification)
classifyButton.place(x=50, y=600)
classifyButton.config(font=font1)

graphButton = Button(main, text="Training Accuracy Graph", command=graph)
graphButton.place(x=430, y=600)
graphButton.config(font=font1)

main.config(bg='turquoise')
main.mainloop()