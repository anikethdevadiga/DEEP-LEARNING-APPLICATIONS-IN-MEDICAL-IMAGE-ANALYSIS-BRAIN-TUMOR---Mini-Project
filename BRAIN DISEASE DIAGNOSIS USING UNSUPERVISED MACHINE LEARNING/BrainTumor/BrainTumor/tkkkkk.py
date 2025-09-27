from tkinter import messagebox, filedialog, Tk, Label, Button, Text, Scrollbar, Frame
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, model_from_json
import pickle
import matplotlib.pyplot as plt
from PIL import Image, ImageTk


def tumor_segmentation(filename):
    # Load the image
    img = cv2.imread(filename)
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to segment the tumor
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image
    segmented_image = img.copy()
    cv2.drawContours(segmented_image, contours, -1, (0, 255, 0), 2)
    
    # Apply edge detection
    edge_image = cv2.Canny(gray, 100, 200)
    
    return segmented_image, edge_image

class TumorDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Identifying Brain Tumor using X-Ray Images")
        self.root.geometry("900x700")
        self.filename = None
        self.accuracy = None
        self.X = []
        self.Y = []
        self.classifier = None
        self.disease = ['No Tumor Detected', 'Tumor Detected']

        self.create_widgets()

    def create_widgets(self):
        # Title
        title_frame = Frame(self.root, bg="darkviolet")
        title_frame.place(relx=0.05, rely=0.02, relwidth=0.9, relheight=0.1)
        title_label = Label(title_frame, text='Identifying Brain Tumor using X-Ray Images', bg="darkviolet", fg="gold", font=('times', 20, 'bold'))
        title_label.place(relx=0.025, rely=0.1, relwidth=0.95, relheight=0.8)

        # Control Frame
        control_frame = Frame(self.root, bg="white")
        control_frame.place(relx=0.05, rely=0.15, relwidth=0.9, relheight=0.25)
        upload_button = Button(control_frame, text="Upload Dataset", command=self.upload_dataset, bg="turquoise", fg="white", font=('times', 14, 'bold'))
        upload_button.place(relx=0.05, rely=0.1, relwidth=0.4, relheight=0.4)
        preprocess_button = Button(control_frame, text="Preprocess Dataset", command=self.dataset_preprocessing, bg="turquoise", fg="white", font=('times', 14, 'bold'))
        preprocess_button.place(relx=0.55, rely=0.1, relwidth=0.4, relheight=0.4)
        train_button = Button(control_frame, text="Train Model", command=self.train_tumor_detection_model, bg="turquoise", fg="white", font=('times', 14, 'bold'))
        train_button.place(relx=0.05, rely=0.55, relwidth=0.4, relheight=0.4)
        classify_button = Button(control_frame, text="Classify Tumor", command=self.tumor_classification, bg="turquoise", fg="white", font=('times', 14, 'bold'))
        classify_button.place(relx=0.55, rely=0.55, relwidth=0.4, relheight=0.4)

        # Result Frame
        result_frame = Frame(self.root, bg="white")
        result_frame.place(relx=0.05, rely=0.42, relwidth=0.9, relheight=0.53)
        self.text = Text(result_frame, wrap="word", bg="lightgrey", font=('times', 12))
        self.text.place(relwidth=1, relheight=1)
        scrollbar = Scrollbar(result_frame, command=self.text.yview)
        scrollbar.place(relx=0.97, rely=0, relheight=1)

    def upload_dataset(self):
        self.filename = filedialog.askdirectory(initialdir=".")
        self.text.delete('1.0', 'end')
        self.text.insert('end', f"{self.filename} loaded\n")

    def dataset_preprocessing(self):
        self.X.clear()
        self.Y.clear()
        if os.path.exists('Model/myimg_data.txt.npy'):
            self.X = np.load('Model/myimg_data.txt.npy')
            self.Y = np.load('Model/myimg_label.txt.npy')
        else:
            for root, dirs, directory in os.walk(os.path.join(self.filename, "no")):
                for name in directory:
                    img = cv2.imread(os.path.join(root, name), 0)
                    img = cv2.resize(img, (128, 128))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(128, 128, 1)
                    self.X.append(im2arr)
                    self.Y.append(0)

            for root, dirs, directory in os.walk(os.path.join(self.filename, "yes")):
                for name in directory:
                    img = cv2.imread(os.path.join(root, name), 0)
                    img = cv2.resize(img, (128, 128))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(128, 128, 1)
                    self.X.append(im2arr)
                    self.Y.append(1)

            self.X = np.asarray(self.X)
            self.Y = np.asarray(self.Y)
            np.save("Model/myimg_data.txt", self.X)
            np.save("Model/myimg_label.txt", self.Y)

        self.text.insert('end', f"Total number of images found in dataset: {len(self.X)}\n")
        self.text.insert('end', f"Total number of classes: {len(set(self.Y))}\n\n")
        self.text.insert('end', "Class labels found in dataset: " + str(self.disease))

    def train_tumor_detection_model(self):
        YY = to_categorical(self.Y)
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        x_train = self.X[indices]
        y_train = YY[indices]

        if os.path.exists('Model/model.json'):
            with open('Model/model.json', "r") as json_file:
                loaded_model_json = json_file.read()
                self.classifier = model_from_json(loaded_model_json)
            self.classifier.load_weights("Model/model_weights.h5")
            self.classifier._make_predict_function()
        else:
            X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
            self.classifier = Sequential()
            self.classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 1), activation='relu'))
            self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
            self.classifier.add(Conv2D(32, (3, 3), activation='relu'))
            self.classifier.add(MaxPooling2D(pool_size=(2, 2)))
            self.classifier.add(Flatten())
            self.classifier.add(Dense(units=128, activation='relu'))
            self.classifier.add(Dense(units=2, activation='softmax'))
            self.classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            hist = self.classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True, verbose=2)
            self.classifier.save_weights('Model/model_weights.h5')
            model_json = self.classifier.to_json()
            with open("Model/model.json", "w") as json_file:
                json_file.write(model_json)
            f = open('Model/history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()

        f = open('Model/history.pckl', 'rb')
        data = pickle.load(f)
        f.close()
        acc = data['accuracy']
        self.accuracy = acc[4] * 100
        self.text.insert('end', '\n\nCNN Brain Tumor Model Generated.\n\n')
        self.text.insert('end', f"CNN Brain Tumor Prediction Accuracy on Test Images: {self.accuracy}\n")

    def tumor_classification(self):
        filename = filedialog.askopenfilename(initialdir="testImages")
        img = cv2.imread(filename, 0)
        img = cv2.resize(img, (128, 128))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 128, 128, 1)
        XX = np.asarray(im2arr)

        predicts = self.classifier.predict(XX)
        cls = np.argmax(predicts)
        if cls == 0:
            messagebox.showinfo("Result", f"Classification Result: {self.disease[cls]}")
        elif cls == 1:
            segmented_image, edge_image = self.tumor_segmentation(filename)
            self.show_image(filename, f"Classification Result: {self.disease[cls]}")
            self.show_image("Tumor Segmented Image", segmented_image)
            self.show_image("Edge Detected Image", edge_image)

    def show_image(self, title, img):
        cv2.imshow(title, img)
        cv2.waitKey(0)

    def tumor_segmentation(self, filename):
        # Your implementation for tumor segmentation
        pass

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = Tk()
    app = TumorDetectionApp(root)
    app.run()
