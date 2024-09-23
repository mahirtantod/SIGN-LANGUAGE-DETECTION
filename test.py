# # import tensorflow as tf
# # from tensorflow.keras.layers import DepthwiseConv2D
# # import cv2
# # from cvzone.HandTrackingModule import HandDetector
# # import numpy as np

# # class CustomDepthwiseConv2D(DepthwiseConv2D):
# #     def __init__(self, *args, **kwargs):
# #         if 'groups' in kwargs:
# #             del kwargs['groups']
# #         super().__init__(*args, **kwargs)

# # # Register the custom layer
# # tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# # # Load the model with the custom layer
# # model = tf.keras.models.load_model(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\keras_model.h5", compile=False)

# # # Load labels
# # with open(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\labels.txt", "r") as f:
# #     labels = [line.strip() for line in f.readlines()]

# # cap = cv2.VideoCapture(0)
# # detector = HandDetector(maxHands=1)

# # while True:
# #     success, img = cap.read()
# #     if not success:
# #         print("Failed to capture image")
# #         continue
    
# #     imgOutput = img.copy()
# #     hands, img = detector.findHands(img)
    
# #     if hands:
# #         hand = hands[0]
# #         x, y, w, h = hand['bbox']
        
# #         # Preprocess the full image
# #         preprocessed = cv2.resize(img, (224, 224))  # Resize to match model input size
# #         preprocessed = preprocessed.astype(np.float32) / 255.0
# #         preprocessed = np.expand_dims(preprocessed, axis=0)

# #         # Make prediction
# #         prediction = model.predict(preprocessed)
# #         index = np.argmax(prediction)

# #         # Draw bounding box and label
# #         cv2.rectangle(imgOutput, (x, y), (x+w, y+h), (0, 255, 0), 2)
# #         cv2.putText(imgOutput, labels[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# #     cv2.imshow('Hand Detection', imgOutput)
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         break

# # cap.release()
# # cv2.destroyAllWindows()
# # import tensorflow as tf
# # from tensorflow.keras.layers import DepthwiseConv2D
# # import cv2
# # import numpy as np
# # import pandas as pd

# # class CustomDepthwiseConv2D(DepthwiseConv2D):
# #     def _init_(self, *args, **kwargs):
# #         if 'groups' in kwargs:
# #             del kwargs['groups']
# #         super()._init_(*args, **kwargs)

# # # Register the custom layer
# # tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# # # Load the model with the custom layer
# # model = tf.keras.models.load_model(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\keras_model.h5", compile=False)

# # # Load labels
# # with open(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\labels.txt", "r") as f:
# #     labels = [line.strip() for line in f.readlines()]

# # # Load and preprocess CSV data
# # def load_and_preprocess_csv(file_path):
# #     df = pd.read_csv(file_path)
# #     pixel_data = df['pixel'].values  # Assuming 'pixel' is the column name
# #     # Reshape and normalize the data
# #     images = np.array([np.fromstring(pixel_string, sep=' ') for pixel_string in pixel_data])
# #     images = images.reshape(-1, 224, 224, 3)  # Reshape to match model input shape
# #     images = images.astype(np.float32) / 255.0
# #     return images

# # # Load your CSV file
# # csv_file_path = r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Data\\archive\\sign_mnist_test\\sign_mnist_test.csv"  # Replace with your CSV file path
# # preprocessed_data = load_and_preprocess_csv(csv_file_path)

# # # Make predictions
# # predictions = model.predict(preprocessed_data)

# # # Process and display results
# # for i, prediction in enumerate(predictions):
# #     index = np.argmax(prediction)
# #     label = labels[index]
# #     confidence = prediction[index]
# #     print(f"Image {i+1}: Predicted Label: {label}, Confidence: {confidence:.2f}")

# # # If you want to visualize some of the images (optional)
# # def display_image(image, label, index):
# #     plt.imshow(image)
# #     plt.title(f"Predicted: {label}")
# #     plt.axis('off')
# #     plt.show()

# # # Display the first few images (you can modify this as needed)
# # import matplotlib.pyplot as plt
# # for i in range(min(5, len(preprocessed_data))):  # Display first 5 or fewer
# #     display_image(preprocessed_data[i], labels[np.argmax(predictions[i])], i)


# import tensorflow as tf
# from tensorflow.keras.layers import DepthwiseConv2D
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np

# class CustomDepthwiseConv2D(DepthwiseConv2D):
#     def __init__(self, *args, **kwargs):
#         if 'groups' in kwargs:
#             del kwargs['groups']
#         super().__init__(*args, **kwargs)

# # Register the custom layer
# tf.keras.utils.get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D

# # Load the model with the custom layer
# model = tf.keras.models.load_model(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\keras_model.h5", compile=False)

# # Load labels
# with open(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\labels.txt", "r") as f:
#     labels = [line.strip() for line in f.readlines()]

# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)

# current_word = ""
# all_words = []

# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to capture image")
#         continue
    
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
    
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
        
#         # Preprocess the full image
#         preprocessed = cv2.resize(img, (224, 224))  # Resize to match model input size
#         preprocessed = preprocessed.astype(np.float32) / 255.0
#         preprocessed = np.expand_dims(preprocessed, axis=0)

#         # Make prediction
#         prediction = model.predict(preprocessed)
#         index = np.argmax(prediction)
#         predicted_label = labels[index]

#         # Update current word or words list
#         if predicted_label.lower() == "space":
#             if current_word:
#                 all_words.append(current_word)
#                 current_word = ""
#         elif predicted_label.lower() == "del":
#             if current_word:
#                 current_word = current_word[:-1]  # Remove last character
#             elif all_words:
#                 current_word = all_words.pop()  # Move last word back to current_word
#         else:
#             current_word += predicted_label

#         # Draw bounding box and label
#         cv2.rectangle(imgOutput, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(imgOutput, predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Display current word and all words
#     cv2.putText(imgOutput, f"Current word: {current_word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv2.putText(imgOutput, f"All words: {' '.join(all_words)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     cv2.imshow('Hand Detection', imgOutput)
    
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

# # Add the last word if it exists
# if current_word:
#     all_words.append(current_word)

# # Combine all words into a single string
# final_string = " ".join(all_words)

# print(f"Final string: {final_string}")

# cap.release()
# cv2.destroyAllWindows()

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
# from tensorflow.keras.models import Model
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import os

# print("TensorFlow version:", tf.__version__)
# print("GPU Available:", tf.config.list_physical_devices('GPU'))

# # Define the model architecture
# def create_model(input_shape, num_classes):
#     inputs = Input(shape=input_shape)
#     x = Conv2D(32, (3, 3), activation='relu')(inputs)
#     x = MaxPooling2D((2, 2))(x)
#     x = Conv2D(64, (3, 3), activation='relu')(x)
#     x = MaxPooling2D((2, 2))(x)
#     x = Flatten()(x)
#     x = Dense(64, activation='relu')(x)
#     outputs = Dense(num_classes, activation='softmax')(x)
    
#     model = Model(inputs=inputs, outputs=outputs)
#     return model

# # Set up data generator
# data_dir = r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Data"
# batch_size = 32
# img_height, img_width = 224, 224

# print("Setting up data generator...")
# datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     horizontal_flip=True
# )

# train_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='sparse',
#     subset='training'
# )

# validation_generator = datagen.flow_from_directory(
#     data_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='sparse',
#     subset='validation'
# )

# print("Data generator setup complete.")
# print(f"Number of classes: {len(train_generator.class_indices)}")
# print(f"Class mapping: {train_generator.class_indices}")

# # Create and compile the model
# print("Creating and compiling model...")
# model = create_model((img_height, img_width, 3), len(train_generator.class_indices))
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# print("Model compilation complete.")

# # Train the model
# print("Starting model training...")
# history = model.fit(
#     train_generator,
#     steps_per_epoch=train_generator.samples // batch_size,
#     validation_data=validation_generator,
#     validation_steps=validation_generator.samples // batch_size,
#     epochs=10
# )

# # Save the trained model
# print("Saving model...")
# model.save(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\trained_model.h5")

# # Save the class names
# class_names = list(train_generator.class_indices.keys())
# with open(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\class_names.txt", "w") as f:
#     for class_name in class_names:
#         f.write(f"{class_name}\n")

# print("Model and class names saved. Starting webcam capture...")

# # Now, use the trained model for predictions
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)

# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to capture image")
#         continue
    
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
    
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
        
#         # Preprocess the full image
#         preprocessed = cv2.resize(img, (img_height, img_width))
#         preprocessed = preprocessed.astype(np.float32) / 255.0
#         preprocessed = np.expand_dims(preprocessed, axis=0)

#         # Make prediction
#         prediction = model.predict(preprocessed)
#         index = np.argmax(prediction)

#         # Draw bounding box and label
#         cv2.rectangle(imgOutput, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(imgOutput, class_names[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     cv2.imshow('Hand Detection', imgOutput)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
# cv2.destroyAllWindows()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

print("TensorFlow version:", tf.__version__)

# Use GPU if available
if tf.config.list_physical_devices('GPU'):
    with tf.device('/GPU:0'):
        # Define the model architecture
        def create_model(input_shape, num_classes):
            inputs = Input(shape=input_shape)
            x = Conv2D(32, (3, 3), activation='relu')(inputs)
            x = MaxPooling2D((2, 2))(x)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D((2, 2))(x)
            x = Flatten()(x)
            x = Dense(64, activation='relu')(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            
            model = Model(inputs=inputs, outputs=outputs)
            return model

        # Set up data generator
        data_dir = r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Data"
        batch_size = 32
        img_height, img_width = 224, 224

        print("Setting up data generator...")
        datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True
        )

        train_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            subset='training'
        )

        validation_generator = datagen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation'
        )

        print("Data generator setup complete.")
        print(f"Number of classes: {len(train_generator.class_indices)}")
        print(f"Class mapping: {train_generator.class_indices}")

        # Create and compile the model
        print("Creating and compiling model...")
        model = create_model((img_height, img_width, 3), len(train_generator.class_indices))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model compilation complete.")

        # Train the model
        print("Starting model training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=10
        )

        # Save the trained model
        print("Saving model...")
        model.save(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\trained_model.h5")

        # Save the class names
        class_names = list(train_generator.class_indices.keys())
        with open(r"D:\\MAJOR_PROJECT\\Sign-Language-detection\\Model\\class_names.txt", "w") as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")

        print("Model and class names saved. Starting webcam capture...")

        # Now, use the trained model for predictions
        cap = cv2.VideoCapture(0)
        detector = HandDetector(maxHands=1)

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to capture image")
                continue
            
            imgOutput = img.copy()
            hands, img = detector.findHands(img)
            
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                
                # Preprocess the full image
                preprocessed = cv2.resize(img, (img_height, img_width))
                preprocessed = preprocessed.astype(np.float32) / 255.0
                preprocessed = np.expand_dims(preprocessed, axis=0)

                # Make prediction
                prediction = model.predict(preprocessed)
                index = np.argmax(prediction)

                # Draw bounding box and label
                cv2.rectangle(imgOutput, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(imgOutput, class_names[index], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Hand Detection', imgOutput)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.destroyAllWindows()
else:
    print("No GPU found. Using CPU for model training and inference.")
    # Proceed with the original code