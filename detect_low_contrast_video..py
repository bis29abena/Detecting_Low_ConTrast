# Usage
# python main.py --video path to videofile

# import the necessary packages
from skimage.exposure import is_low_contrast
import cv2 as cv
import numpy as np
import imutils
import argparse

# Construct an argument parser to parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to video file", required=True, type=str)
ap.add_argument("-t", "--thresh", help="Threshold value for low contrast", default=0.35, type=float)
args = vars(ap.parse_args())

# grab a pointer to the input video stream
print(f"[INFO] Loading video stream")
vs = cv.VideoCapture(args["video"] if args["video"] else 0)

# loop over the frames of the video
while True:
    # read a frame fro the video stream
    (grabbed, frame) = vs.read()

    # if the video frame was not grabbed then we have reached the end of the video
    if not grabbed:
        print("[INFO] no frame read from stream exiting ")
        break

    # resize the frame, covert it to grayscale blur it and the perform edge detection
    frame = imutils.resize(frame, width=450)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    edged = cv.Canny(blurred, 30, 200)

    # initialize the text, object_text and color to indicate that the current frame is not low contrast
    text = "Low Contrast: NO"
    object_text = "Object Detected: Yes"
    color = (0, 255, 0)

    # check to see if the frame is of low contrast and update
    # the text, object text and color
    if is_low_contrast(edged, args["thresh"]):
        text = "Low Contrast: Yes"
        object_text = "Object Detected: No"
        color = (0, 0, 255)

    else:
        # find the contours in the edge map and fid the target one
        # we assume is the outline of our color correction card
        cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv.contourArea)

        # draw the target contour on the image
        cv.drawContours(frame, [c], -1, (0, 255, 0), 2)

    # draw the text on the output image
    cv.putText(frame, text, (5, 25), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv.putText(frame, object_text, (5, 50), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # stack the output frame and the edged map together
    output = np.dstack([edged] * 3)
    output = np.hstack([frame, output])

    # show the output to our screen
    cv.imshow("Output", output)
    key = cv.waitKey(1) & 0xFF

    # if the q key is pressed break from the loop
    if key == ord("q"):
        break
cv.destroyAllWindows()


