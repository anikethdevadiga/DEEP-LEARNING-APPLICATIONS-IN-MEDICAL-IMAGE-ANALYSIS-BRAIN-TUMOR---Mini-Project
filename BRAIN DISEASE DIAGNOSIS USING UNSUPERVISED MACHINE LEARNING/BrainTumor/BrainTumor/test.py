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
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle
from sklearn import metrics
import ftplib
from tkinter import ttk

main = tkinter.Tk()
main.title("Identifying Brain Tumor using X-Ray Images") # designing main screen
main.geometry("1300x1200")

global filename
global accuracy
X = []
Y = []
global classifier
disease = ['No Tumor Detected', 'Tumor Detected']

with open('Model/segmented_model.json', "r") as json_file:
    loaded_model_json = json_file.read()
    segmented_model = model_from_json(loaded_model_json)
segmented_model.load_weights("Model/segmented_weights.h5")

def edgeDetection():
    img = cv2.imread('myimg.png')
    orig = cv2.imread('test1.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    min_area = 0.95 * 180 * 35
    max_area = 1.05 * 180 * 35
    result = orig.copy()
    for c in contours:
        area = cv2.contourArea(c)
        cv2.drawContours(result, [c], -1, (0, 0, 255), 10)
        if min_area < area < max_area:
            cv2.drawContours(result, [c], -1, (0, 255, 255), 10)
    return result

def tumorSegmentation(filename):
    global segmented_model
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, 64, 64, 1)
    img = (img - 127.0) / 127.0
    preds = segmented_model.predict(img)
    preds = preds[0]
    orig = cv2.imread(filename, 0)
    orig = cv2.resize(orig, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("test1.png", orig)    
    segmented_image = cv2.resize(preds, (300, 300), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("myimg.png", segmented_image * 255)
    edge_detection = edgeDetection()
    return segmented_image * 255, edge_detection

def uploadDataset(): # function to upload dataset
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def datasetPreprocessing():
    global X
    global Y
    X.clear()
    Y.clear()
    if os.path.exists('Model/myimg_data.txt.npy'):
        X = np.load('Model/myimg_data.txt.npy')
        Y = np.load('Model/myimg_label.txt.npy')
    else:
        for root, dirs, directory in os.walk(filename + "/no"):
            for name in directory:
                img = cv2.imread(filename + "/no/" + name, 0) # reading images
                img = cv2.resize(img, (128, 128)) # resizing images
                im2arr = np.array(img) # extract features from images
                im2arr = im2arr.reshape(128, 128, 1)
                X.append(im2arr)
                Y.append(0)

        for root, dirs, directory in os.walk(filename + "/yes"):
            for name in directory:
                img = cv2.imread(filename + "/yes/" + name, 0)
                img = cv2.resize(img, (128, 128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(128, 128, 1)
                X.append(im2arr)
                Y.append(1)
                
        X = np.asarray(X)
        Y = np.asarray(Y)            
        np.save("Model/myimg_data.txt", X)
        np.save("Model/myimg_label.txt", Y)
    text.insert(END, "Total number of images found in dataset : " + str(len(X)) + "\n")
    text.insert(END, "Total number of classes : " + str(len(set(Y))) + "\n\n")
    text.insert(END, "Class labels found in dataset : " + str(disease))       

def trainTumorDetectionModel():
    global accuracy
    global classifier
    
    YY = to_categorical(Y)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    x_train = X[indices]
    y_train = YY[indices]

    if os.path.exists('Model/model.json'):
        with open('Model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        classifier.load_weights("Model/model_weights.h5")           
    else:
        X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
        classifier = Sequential() 
        classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Conv2D(32, (3, 3), activation='relu'))
        classifier.add(MaxPooling2D(pool_size=(2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(units=128, activation='relu'))
        classifier.add(Dense(units=2, activation='softmax'))
        classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
        classifier.save_weights('Model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("Model/model.json", "w") as json_file:
            json_file.write(model_json)
        with open('Model/history.pckl', 'wb') as f:
            pickle.dump(hist.history, f)
    with open('Model/history.pckl', 'rb') as f:
        data = pickle.load(f)
    acc = data['accuracy']
    accuracy = acc[-1] * 100
    text.insert(END, '\n\nCNN Brain Tumor Model Generated.\n\n')
    text.insert(END, "CNN Brain Tumor Prediction Accuracy on Test Images : " + str(accuracy) + "\n")

def tumorClassification():
    filename = filedialog.askopenfilename(initialdir="testImages")
    img = cv2.imread(filename, 0)
    img = cv2.resize(img, (128, 128))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1, 128, 128, 1)
    XX = np.asarray(im2arr)
        
    predicts = classifier.predict(XX)
    cls = np.argmax(predicts)
    if cls == 0:
        img = cv2.imread(filename)
        img = cv2.resize(img, (800, 500))
        cv2.putText(img, 'Classification Result : ' + disease[cls], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : ' + disease[cls], img)
        cv2.waitKey(0)
    if cls == 1:
        segmented_image, edge_image = tumorSegmentation(filename)
        img = cv2.imread(filename)
        img = cv2.resize(img, (800, 500))
        cv2.putText(img, 'Classification Result : ' + disease[cls], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Classification Result : ' + disease[cls], img)
        cv2.imshow("Tumor Segmented Image", segmented_image)
        cv2.imshow("Edge Detected Image", edge_image)
        cv2.waitKey(0)

def graph():
    with open('Model/history.pckl', 'rb') as f:
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

font = ('times', 16, 'bold')
title = Label(main, text='Identifying Brain Tumor using X-Ray Images')
title.config(bg='darkviolet', fg='gold')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0, y=5)
font1 = ('times', 14, 'bold')
upload_button = Button(main, text="Upload Dataset", command=uploadDataset)
upload_button.place(x=50, y=100)
upload_button.config(font=font1)

preprocess_button = Button(main, text="Dataset Preprocessing", command=datasetPreprocessing)
preprocess_button.place(x=300, y=100)
preprocess_button.config(font=font1)

train_button = Button(main, text="Train Tumor Detection Model", command=trainTumorDetectionModel)
train_button.place(x=600, y=100)
train_button.config(font=font1)

classify_button = Button(main, text="Classify Tumor in Test Image", command=tumorClassification)
classify_button.place(x=950, y=100)
classify_button.config(font=font1)

graph_button = Button(main, text="CNN Training Accuracy & Loss Graph", command=graph)
graph_button.place(x=1300, y=100)
graph_button.config(font=font1)

font2 = ('times', 12, 'bold')
text = Text(main, height=30, width=150)
text.place(x=50, y=150)
text.config(font=font2)

main.config(bg='darkviolet')
main.mainloop()
