# import necessary OpenCV modules
import cv2
import numpy as np

# read video file
cap = cv2.VideoCapture("videos/vid-0-12.mp4")

# create old and new frames
old_frame = None
new_frame = None

matches = []
count = 0
while True:
    # read new frame from video
    success, new_frame = cap.read()
    

    if not success:
        break
    
    new_frame = new_frame[int(new_frame.shape[0]/3) : 2 * int(new_frame.shape[0]/3), :]

    if old_frame is None:
        old_frame = new_frame
        continue

    # convert frames to grayscale
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)

    # detect corners in old frame
    old_corners = cv2.goodFeaturesToTrack(old_gray, maxCorners=500, qualityLevel=0.01, minDistance=10)

    # calculate optical flow for corners using Lucas-Kanade algorithm
    new_corners, status, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_corners, None)

    # eliminate corners that are moving outside of the object
    for i in range(len(new_corners)):
        if np.linalg.norm(new_corners[i] - old_corners[i]) < 50:
            new_corners[i] = np.array([0, 0], dtype=np.float32)
     
    
   

    # convert new corners to keypoints
    new_keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in new_corners]

    # draw keypoints on new frame
    new_frame_with_keypoints = cv2.drawKeypoints(new_frame, new_keypoints, None, color=(0, 255, 0))
    #cv2.imwrite(f"Frame with keypoints{count}.png", new_frame_with_keypoints)
    #count+=1
    # display new frame with keypoints
    cv2.imshow("Frame with keypoints", new_frame_with_keypoints)
    cv2.waitKey(1)

    # update old frame and continue to next iteration
    old_frame = new_frame
