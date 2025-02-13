import cv2
import numpy as np
import os

def countCoins(imagePath, outputDir="outputImages"):

# IMAGE PREPROCESSING
# --------------------
    # load the image
    originalImage = cv2.imread(imagePath)
    image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]

    # scale the image down to control the contour area correctly
    # determine the scaling factor
    scale = min(1024 / width, 1024 / height)

    # If the image is already within size, return as is
    if scale < 1:        
        newWidth = int(width * scale)
        newHeight = int(height * scale)
        image = cv2.resize(image, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
        originalImage = cv2.resize(originalImage, (newWidth, newHeight), interpolation=cv2.INTER_AREA)

    # apply Gaussian filter to reduce noise
    image = cv2.GaussianBlur(image, (7, 7), 0)
# FEATURE EXTRACTION
# ------------------
    # apply sobel for edge detection
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradientMagnitude = cv2.magnitude(sobelX, sobelY)
    gradientMagnitude = np.uint8(gradientMagnitude)

    # use another Gaussian filter to smooth detected edges else we get noise
    gradientMagnitude = cv2.GaussianBlur(gradientMagnitude, (7, 7), 0)

    # apply thresholding to get clean binary image
    _, binaryImage = cv2.threshold(gradientMagnitude, 60, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# SUBJECT FINDING
# ---------------
    # find continuous contours in the binary image
    contours, _ = cv2.findContours(binaryImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # remove small contour areas to eliminate noisey patterns
    minArea = 500
    filteredContours = [cnt for cnt in contours if cv2.contourArea(cnt) > minArea]

    # for display purpose draw contours on the original image
    for contour in filteredContours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(originalImage, (x, y), (x + w, y + h), (0, 255, 0), 2)

# SAVING THE RESULTS
# ------------------
    # generate output filenames dynamically
    imageName = os.path.basename(imagePath).split('.')[0]
    os.makedirs(outputDir, exist_ok=True)

    cv2.imwrite(f"{outputDir}/{imageName}_gradient.jpg", gradientMagnitude)
    cv2.imwrite(f"{outputDir}/{imageName}_binary.jpg", binaryImage)
    cv2.imwrite(f"{outputDir}/{imageName}_contours.jpg", originalImage)

    print(f"Number of coins in for {imagePath} : ", len(filteredContours))
    print(f"Processed results for {imagePath}\n")

# image directories
inputDir = "inputImages"
outputDir = "outputImages"

# go through all image files in the input directory
for filename in os.listdir(inputDir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        imagePath = os.path.join(inputDir, filename)
        countCoins(imagePath, outputDir)
