import numpy as np 
import cv2

# load image
image = cv2.imread('leaf.jpg')

# Check if image is None
if image is None:
    print("UPLOAD A POTATO LEAF IMAGE")
else:
    # convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # set lower and upper color limits for potato leaves
    low_val = (30, 50, 50)  # Example values for potato leaves
    high_val = (90, 255, 255)  # Example values for potato leaves

    # Threshold the HSV image 
    mask = cv2.inRange(hsv, low_val, high_val)

    # remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((8,8),dtype=np.uint8))

    # apply mask to original image
    result = cv2.bitwise_and(image, image, mask=mask)

    # show images
    cv2.imshow("Result", result)
    cv2.imshow("Mask", mask)
    cv2.imshow("Image", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
