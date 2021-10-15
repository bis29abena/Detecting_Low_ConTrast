# Usage
# python main.py --image imagefile

# import the necessary packages
from skimage.exposure import is_low_contrast
from imutils.paths import list_images
import cv2 as cv
import imutils
import argparse

# Construct an argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to input images", required=True, type=str)
ap.add_argument("-t", "--thresh", help="Threshold value for low contrast", default=0.35, type=float)
args = vars(ap.parse_args())

# grab the paths to the input_images
imagePaths = sorted(list(list_images(args["image"])))

# loop over the imagePaths
for (i, imagePath) in enumerate(imagePaths):
    # load the input image from the disk, resize it and convert it to grayscale
    print(f"[INFO] Processing image {i +1}/{len(imagePaths)}")
    image = cv.imread(imagePath)
    image = imutils.resize(image, width=600)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # blur the image slightly and perform edge detection
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blur, 30, 150)

    # initialize the text and color and to indicate that the input image
    # is for low contrast
    text = "Low Contrast: No"
    object_text = "Object Detected"
    color = (0, 255, 0)

    # check to see if the image is of low contrast
    if is_low_contrast(gray, args["thresh"]):
        # update the text and color
        text = "Low Contrast: Yes"
        object_text = "No Object"
        color = (0, 0, 255)

    # image is not low contrast so we continue to process it
    else:
        # find the contours in the edge map and fid the target one
        # we assume is the outline of our color correction card
        cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # draw the target contour on the image
        cv.drawContours(image, [c], -1, (0, 255, 0), 2)

    # draw the text on the output image
    cv.putText(image, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.putText(image, object_text, (5, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # show the output image and the edged map
    cv.imshow("Image", image)
    cv.imshow("Edged", edged)
    cv.waitKey(0)