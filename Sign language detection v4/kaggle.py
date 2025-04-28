import cv2
import os
import numpy as np
import math

# Path to the folder containing the Kaggle dataset
dataset_path = "C:\\Users\\SOLOMON\\Downloads\\SigNN Character Database"
output_folder = "C:\\Users\\SOLOMON\\Downloads\\SigNN Character Database\\Preprocessed"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Initialize hand detector
from cvzone.HandTrackingModule import HandDetector
detector = HandDetector(maxHands=1)

imgSize = 300  # Final image size after processing
offset = 20  # Extra space around the hand

# Loop through the dataset directories (A, B, C, etc.)
for label in os.listdir(dataset_path):
    label_folder = os.path.join(dataset_path, label)

    if os.path.isdir(label_folder):  # Process only folders
        # Create folder for preprocessed images
        label_output_folder = os.path.join(output_folder, label)
        if not os.path.exists(label_output_folder):
            os.makedirs(label_output_folder)

        # Loop through the images in the folder
        for img_name in os.listdir(label_folder):
            img_path = os.path.join(label_folder, img_name)
            
            # Read the image
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Detect hands in the image
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
                    continue

                # Aspect ratio for resizing
                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize[:imgSize, :wCal]
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize[:hCal, :imgSize]

                # Save the preprocessed image
                output_img_path = os.path.join(label_output_folder, img_name)
                cv2.imwrite(output_img_path, imgWhite)

        print(f"Processed {label} images.")

print("Processing complete!")
