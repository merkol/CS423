import cv2
import numpy as np

# Reading the first video
vidcap = cv2.VideoCapture('videos/vid-0-12.mp4')

# Read the first frame
success, image = vidcap.read()

# Loop through the frames
while success:
  # Convert to grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Corner Detection
    corners = cv2.cornerHarris(img_gray, blockSize=2, ksize=3, k=0.04)
    image[corners > 0.01 * corners.max()] = [0, 0, 255]
    
    # Keypoint detection over threshold
    keypoints = np.argwhere(corners > 0.01 * corners.max())
    keypoints = [cv2.KeyPoint(float(x[1]), float(x[0]), 13) for x in keypoints]
    
    # Create a SIFT object
    sift = cv2.SIFT_create()
    # Detect keypoints and compute the descriptors
    sift.compute(image, keypoints)
    # Draw the keypoints on the image
    img = cv2.drawKeypoints(image, keypoints, image)

    cv2.imshow('dst', image)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    
    # Read the next frame
    success, image = vidcap.read()

    # Print a message
    print('Read a new frame: ', success)
    
