import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)  # Change to 1 or 2 if using external camera
detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder = "D:\Sign language detection v4\Data\D"

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Exiting.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Ensure crop coordinates are within image boundaries
        y1 = max(0, y - offset)
        y2 = min(y + h + offset, img.shape[0])
        x1 = max(0, x - offset)
        x2 = min(x + w + offset, img.shape[1])
        imgCrop = img[y1:y2, x1:x2]

        imgCropShape = imgCrop.shape

        if imgCropShape[0] == 0 or imgCropShape[1] == 0:
            print("Invalid crop. Skipping this frame.")
            continue

        aspectratio = h / w

        if aspectratio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize[:imgSize, :wCal]
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize[:hCal, :imgSize]

        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image saved: {counter}")

cap.release()
cv2.destroyAllWindows()
