import cv2
import numpy as np
import os
import pickle
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils.np_utils import to_categorical
from keras.layers import MaxPooling2D, Dense, Dropout, Activation, Flatten, Conv2D
from keras.models import Sequential, model_from_json
import matplotlib.pyplot as plt
import zipfile
import shutil
import imutils

# Tkinter related imports are not needed anymore

st.title("Identifying Brain Tumor using X-Ray Images")

global filename
global accuracy
X = []
Y = []
global classifier
disease = ['No Tumor Detected', 'Tumor Detected']

# Integration with Streamlit
uploaded_file = st.file_uploader("Upload Brain Tumor Dataset (zip file)", type="zip")
if uploaded_file:
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall("brain_tumor_dataset")

        # Now, you can proceed with the data preprocessing, training, and classification steps

        # Data preprocessing function
        def datasetPreprocessing():
            global X
            global Y
            X.clear()
            Y.clear()
            
            dataset_path = "brain_tumor_dataset"  # Path to the extracted dataset directory
            
            if not os.path.exists(dataset_path):
                st.error("Dataset directory not found.")
                return
            
            for root, dirs, files in os.walk(dataset_path):
                for subdir in dirs:
                    subdir_path = os.path.join(root, subdir)
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        try:
                            img = cv2.imread(file_path, 0)
                            if img is None:
                                st.warning(f"Failed to load image: {file_path}")
                                continue
                            img = cv2.resize(img, (128, 128))
                            im2arr = np.array(img)
                            im2arr = im2arr.reshape(128, 128, 1)
                            X.append(im2arr)
                            Y.append(1 if subdir == "yes" else 0)
                        except Exception as e:
                            st.error(f"Error processing image {file}: {e}")

            if not X or not Y:
                st.error("No images found or error occurred during processing.")
                return
            
            X = np.asarray(X)
            Y = np.asarray(Y)
            
            st.write("Total number of images found in dataset:", len(X))
            st.write("Total number of classes:", len(set(Y)))
            st.write("Class labels found in dataset:", disease)

        # CNN Model training function
        def trainTumorDetectionModel():
            global accuracy
            global classifier

            YY = to_categorical(Y)

            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)

            x_train = X[indices]
            y_train = YY[indices]

            # Ensure Model directory exists
            if not os.path.exists('Model'):
                os.makedirs('Model')

            # Model creation and training
            if os.path.exists('Model/model.json'):
                with open('Model/model.json', "r") as json_file:
                    loaded_model_json = json_file.read()
                    classifier = model_from_json(loaded_model_json)

                classifier.load_weights("Model/model_weights.h5")
                # classifier._make_predict_function()
            else:
                X_trains, X_tests, y_trains, y_tests = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
                classifier = Sequential()
                classifier.add(Conv2D(32, 3, 3, input_shape=(128, 128, 1), activation='relu'))
                classifier.add(MaxPooling2D(pool_size=(2, 2)))
                classifier.add(Conv2D(32, 3, 3, activation='relu'))
                classifier.add(MaxPooling2D(pool_size=(2, 2)))
                classifier.add(Flatten())
                classifier.add(Dense(output_dim=128, activation='relu'))
                classifier.add(Dense(output_dim=2, activation='softmax'))
                classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                hist = classifier.fit(x_train, y_train, batch_size=16, epochs=10, validation_split=0.2, shuffle=True,
                                      verbose=2)
                classifier.save_weights('Model/model_weights.h5')
                model_json = classifier.to_json()
                with open("Model/model.json", "w") as json_file:
                    json_file.write(model_json)
                f = open('Model/history.pckl', 'wb')
                pickle.dump(hist.history, f)
                f.close()

            f = open('Model/history.pckl', 'rb')
            data = pickle.load(f)
            f.close()
            acc = data['accuracy']
            accuracy = acc[4] * 100
            st.write('\n\nCNN Brain Tumor Model Generated. See black console to view layers of CNN\n\n')
            st.write("CNN Brain Tumor Prediction Accuracy on Test Images:", accuracy)

        # Function for tumor classification
        def tumorClassification():
            filename = st.file_uploader("Upload Image for Classification", type=["png", "jpg", "jpeg"])
            if filename is not None:
                img = cv2.imdecode(np.fromstring(filename.read(), np.uint8), 1)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (128, 128))
                im2arr = np.array(img)
                im2arr = im2arr.reshape(1, 128, 128, 1)
                XX = np.asarray(im2arr)

                predicts = classifier.predict(XX)
                cls = np.argmax(predicts)
                if cls == 0:
                    st.write("Classification Result: No Tumor Detected")
                elif cls == 1:
                    st.write("Classification Result: Tumor Detected")
                else:
                    st.write("Invalid Classification Result")

        if st.button("Dataset Preprocessing & Features Extraction"):
            datasetPreprocessing()

        if st.button("Trained CNN Brain Tumor Detection Model"):
            trainTumorDetectionModel()

        if st.button("Brain Tumor Segmentation & Classification"):
            tumorClassification()
