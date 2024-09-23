import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
counter = 0

folder = "D:\MAJOR_PROJECT\Sign-Language-detection\Data\Love you"

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    # Inside your main loop
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
        imgCropShape = imgCrop.shape


        # Check if imgCrop is valid before proceeding
        if imgCrop.size > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize

            # Display only if imgCrop is valid
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        else:
            print("No valid image to crop.")
    else:
        print("No hands detected.")

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)

# import cv2
# from cvzone.HandTrackingModule import HandDetector
# import numpy as np
# import time

# # Set up video capture and hand detector
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# output_file = "D:\MAJOR PROJE CT\Sign-Language-detection\Data\Love you"

# # Define codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use different codecs like 'MJPG'
# fps = 24  # Frame rate
# frame_width = int(cap.get(3))  # Width of the frame
# frame_height = int(cap.get(4))  # Height of the frame

# # Create VideoWriter object
# out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

# while True:
#     success, img = cap.read()
#     if not success:
#         print("Failed to capture image.")
#         break

#     hands, img = detector.findHands(img)

#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     # Write the frame to the video file
#     out.write(img)

#     # Display the resulting frame
#     cv2.imshow('Image', img)

#     # Break the loop on 'q' key press
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release everything when done
# cap.release()
# out.release()
# cv2.destroyAllWindows()
