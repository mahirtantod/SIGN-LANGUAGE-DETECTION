import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

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
data_dir = r"D:\\projects\\sign_language\\new major project\\Data"
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
    epochs=500
)

# Save the trained model
print("Saving model...")
model.save(r"D:\\projects\\sign_language\\new major project\\Model\\trained_model.h5")

# Save the class names
class_names = list(train_generator.class_indices.keys())
with open(r"D:\\projects\\sign_language\\new major project\\Model\\class_names.txt", "w") as f:
    for class_name in class_names:
        f.write(f"{class_name}\n")

print("Model and class names saved. Starting webcam capture...")

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
#         # prediction = Model.predict(preprocessed)
#         prediction = model.predict(x=preprocessed)
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
