# import necessary OpenCV modules
import cv2
import numpy as np

# read video file
cap = cv2.VideoCapture("videos/vid-0-12.mp4")

# sift object
sift = cv2.SIFT_create()

# create BFMatcher object 
bf = cv2.BFMatcher()

# create old and new frames
old_frame = None
new_frame = None


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

    # old keypoints for SIFT
    old_keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in old_corners]
    
    # calculate optical flow for corners using Lucas-Kanade algorithm
    new_corners, status, err = cv2.calcOpticalFlowPyrLK(old_gray, new_gray, old_corners, None)

    # eliminate corners that are moving outside of the object
    for i in range(len(new_corners)):
        if np.linalg.norm(new_corners[i] - old_corners[i]) < 50:
            new_corners[i] = np.array([0, 0], dtype=np.float32)
     
    # convert new corners to keypoints
    new_keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in new_corners]
    
    # SIFT on previous frame
    _ , old_descriptors = sift.compute(old_gray, old_keypoints)
    
    # SIFT on new frame
    _ , new_descriptors = sift.compute(new_gray, new_keypoints)
    
    # Match descriptors
    matches = bf.knnMatch(old_descriptors, new_descriptors, k=2)
    
    # Filter matches using the Lowe ratio
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Use the matches to find the transformation between the frames
    if len(good_matches) > 3:
        src_pts = np.float32([old_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([new_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Use RANSAC to find the transformation and eliminate outliers
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        
        # Apply the transformation to align the frames
        h, w = old_gray.shape
        aligned_frame = cv2.warpPerspective(new_frame, M, (w, h))

   
    # draw matches
    img_matches = cv2.drawMatches(old_frame, old_keypoints, new_frame, new_keypoints, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(1)  

    # draw keypoints on new frame
    #new_frame_with_keypoints = cv2.drawKeypoints(new_frame, new_keypoints, None, color=(0, 255, 0))
    #cv2.imwrite(f"Frame with keypoints{count}.png", new_frame_with_keypoints)
    #count+=1
    
    # display new frame with keypoints
    #cv2.imshow("Frame with keypoints", new_frame_with_keypoints)
    #cv2.waitKey(1)

    # update old frame and continue to next iteration
    old_frame = new_frame
