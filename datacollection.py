import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize variables
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300

# Create base directory for dataset
base_folder = "D:\\MAJOR_PROJECT\\Sign-Language-detection\\Data"

# Create folders for each letter if they don't exist
for letter in range(ord('A'), ord('Z')+1):
    letter_folder = os.path.join(base_folder, chr(letter))
    os.makedirs(letter_folder, exist_ok=True)

# Variables for tracking progress
current_letter = 'A'
counter = 0
total_letters = 26
images_per_letter = 3000

def switch_to_next_letter():
    global current_letter, counter
    if current_letter < 'Z':
        current_letter = chr(ord(current_letter) + 1)
        counter = 0
        print(f"\nSwitching to letter {current_letter}")
        print("Press 's' to start capturing images for this letter")
    else:
        print("\nData collection completed for all letters!")
        cap.release()
        cv2.destroyAllWindows()
        exit()

print(f"Starting with letter {current_letter}")
print("Press 's' to capture images")
print("Press 'n' to switch to next letter")
print("Press 'q' to quit")

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        
        if imgCrop.size > 0:
            aspectRatio = h / w
            
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
    
    # Display current progress
    progress_text = f"Letter: {current_letter} | Images: {counter}/{images_per_letter}"
    cv2.putText(img, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0, 255, 0), 2)
    
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    
    if key == ord("s"):
        if counter < images_per_letter:
            current_folder = os.path.join(base_folder, current_letter)
            cv2.imwrite(f'{current_folder}/Image_{time.time()}.jpg', imgWhite)
            counter += 1
            print(f"Captured image {counter}/{images_per_letter} for letter {current_letter}")
            
            if counter >= images_per_letter:
                print(f"\nCompleted capturing {images_per_letter} images for letter {current_letter}")
                print("Press 'n' to continue to next letter or 'q' to quit")
    
    elif key == ord("n"):
        switch_to_next_letter()
    
    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
