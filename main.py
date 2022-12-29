# import necessary OpenCV modules
import cv2
import numpy as np

# read video file
cap = cv2.VideoCapture("videos/vid-2-12.mp4")

# sift object
sift = cv2.SIFT_create()

# create BFMatcher object 
bf = cv2.BFMatcher()

# create old and new frames
old_frame = None
new_frame = None

right_counter = 0
left_counter = 0

dir_list = []
count = 0
while True:
    
    # read new frame from video
    success, new_frame = cap.read()
    
    if not success:
        break
    
    new_frame = new_frame[0  : int(new_frame.shape[0]/3), :]

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
        if np.linalg.norm(new_corners[i] - old_corners[i]) < 20:
            new_corners[i] = np.array([0, 0], dtype=np.float32)
            old_corners[i] = np.array([0, 0], dtype=np.float32)
     
    # convert new corners to keypoints
    new_keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in new_corners]
    
    # old keypoints for SIFT
    old_keypoints = [cv2.KeyPoint(x=c[0][0], y=c[0][1], size=10) for c in old_corners]
    
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

        M, mask = cv2.estimateAffine2D(src_pts, dst_pts, method = cv2.RANSAC)
        
        # Determine the object's moving direction
        if M[0, 0] > 0 and M[1, 1] > 0:
            print("The object is moving down and to the right")
        elif M[0, 0] < 0 and M[1, 1] < 0:
            print("The object is moving up and to the left")
        elif M[0, 0] > 0 and M[1, 1] < 0:
            print("The object is moving down and to the left")
        elif M[0, 0] < 0 and M[1, 1] > 0:
            print("The object is moving up and to the right")
        else:
            print("The object is not rotating")

        # Determine the object's translation
        translation_x = M[0, 2]
        translation_y = M[1, 2]
      
        print(f"The object is moving by {translation_x} units along the x-axis and {translation_y} units along the y-axis")
        
        if (translation_y < 5 and translation_y > -5):
            if translation_x > 0:
                right_counter += 1
            elif translation_x < 0:
                left_counter += 1 
            
        print(f"Right counter : {right_counter}")
        print(f"Left counter : {left_counter}")

        
        # Apply the transformation to align the frames
        h, w = old_gray.shape
        aligned_frame = cv2.warpAffine(new_frame, M, (w, h)) 
        
        # compute rotation matrix from affine transformation
        M_rot = cv2.getRotationMatrix2D(center=(0, 0), angle = np.rad2deg(np.arctan2(M[1, 0], M[0, 0])), scale = 1)

        # compute angle of rotation from rotation matrix
        angle = np.rad2deg(np.arctan2(M_rot[1, 0], M_rot[0, 0]))
    
        # determine direction of fish passage based on angle of rotation
       
        if angle > 0:
            direction = "right"
        elif angle < 0:
            direction = "left"
        else:
            direction = "unknown"
        
        dir_list.append(direction)
    
    # draw matches
    img_matches = cv2.drawMatches(old_frame, old_keypoints, new_frame, new_keypoints, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)  
    
    # draw keypoints on new frame
    #new_frame_with_keypoints = cv2.drawKeypoints(new_frame, new_keypoints, None, color=(0, 255, 0))
    #cv2.imwrite(f"Frame with keypoints{count}.png", new_frame_with_keypoints)
    #count+=1
    
    # display new frame with keypoints
    #cv2.imshow("Frame with keypoints", new_frame_with_keypoints)
    #cv2.waitKey(1)

    # update old frame and continue to next iteration
    old_frame = new_frame

