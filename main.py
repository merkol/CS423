# Mert Erkol S017789 Department of Computer Science 

import cv2
import numpy as np
from sklearn.cluster import MeanShift


# read video file
cap = cv2.VideoCapture("videos/vid-5-18.mp4")

# sift object
sift = cv2.SIFT_create()

# create BFMatcher object 
bf = cv2.BFMatcher()

# create old and new frames
old_frame = None
new_frame = None

# frame counter
frame_count = 0

# Fish counter
fish_count = 0

# Fish detection
fish_detected = {0: '', 1: '', 2: ''}

# Fish Exited or not
fish_exited = {0: '', 1: '', 2: ''}


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
        if np.linalg.norm(new_corners[i] - old_corners[i]) < 25:
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

    # After filtering the matches using the Lowe ratio, create a matrix of the keypoint coordinates optimum parameters (200, 25)
    def get_cluster_points(keypoints, bandwidth=200, cluster_threshold=25):
        points = np.zeros((len(good_matches), 2))
        for i, match in enumerate(good_matches):
            old_keypoint = keypoints[match.queryIdx]
            points[i, :] = (old_keypoint.pt[0], old_keypoint.pt[1])
            
        # Initialize the MeanShift algorithm with the points
        clustering = MeanShift(bandwidth=bandwidth).fit(points)

        # Extract the cluster labels for each point
        labels = clustering.labels_
        
        # Draw the clusters on the image
        for i, point in enumerate(points):
            x, y = point
            x = x.astype(int)
            y = y.astype(int)
            cluster = labels[i]
            count_val = (labels == cluster).sum()
            if count_val > cluster_threshold:
                cv2.circle(new_frame, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(new_frame, str(cluster), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        # Create a new list of points and labels for clusters with more than 25 points
        filtered_points = [point for i, point in enumerate(points) if (labels == labels[i]).sum() > cluster_threshold]
        filtered_labels = [label for i, label in enumerate(labels) if (labels == labels[i]).sum() > cluster_threshold]

        # Update the points and labels arrays with the filtered lists
        points = np.array(filtered_points)
        labels = np.array(filtered_labels)

        # Find the unique labels
        unq = np.unique(labels).size
        
        # Find the cluster centers
        centers = clustering.cluster_centers_
        
        # Deleting the deleted clusters center points 
        if unq == 0:
            centers = np.array([])
        elif unq != centers.shape[0]:
            sh = centers.shape[0] - unq
            centers = np.delete(centers, np.s_[-sh:], axis=0)
            
        
        for i,j in centers:
            cv2.circle(new_frame, (int(i), int(j)), 5, (0, 0, 255), -1)
            cv2.putText(new_frame, "Center", (int(i), int(j)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
    
        return points, labels, centers
    
    datapoints = get_cluster_points(old_keypoints)    
    
    
    # Use the matches to find the transformation between the frames
    if len(good_matches) > 3 and datapoints[0].size != 0:
        
        for i in range(np.unique(datapoints[1]).size):
            src_pts = np.float32([old_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([new_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            ## Filtering the points from the clusters
            cluster_condition = np.where(datapoints[1] == i)[0]
            clustered_points = datapoints[0][cluster_condition]
            result = (src_pts[:, None] == clustered_points).all(-1).any(-1)
            indexes = []
            for it,j in enumerate(result):
                if j == True:
                    indexes.append(it)
            index = np.isin(np.arange(dst_pts.shape[0]), indexes)
            dst_pts = dst_pts[index]

        
        
            # Estimate the affine transformation between the frames
            M, mask = cv2.estimateAffine2D(clustered_points, dst_pts, method = cv2.RANSAC)
        
            # Determine the object's translation
            translation_x = M[0, 2]
            translation_y = M[1, 2]
        
            # Draw the object's trajectory
            if i == 0:
                color = (255, 255, 0)
            elif i == 1:
                color = (0, 255, 0)
            
            if fish_count < len(datapoints[2]) and fish_detected[i] != 'Detected':
                st_point = (int(datapoints[2][fish_count][0]), int(datapoints[2][fish_count][1]))
                fish_detected[fish_count] = 'Detected'
                fish_count += 1

                
            end_point = int(translation_x + st_point[0]), int(translation_y + st_point[1])
            new_frame = cv2.line(new_frame, st_point, end_point, color,3)
            st_point = end_point
            if end_point[0] > 1364 and fish_exited[i] != 'Exited':
                print(f"Fish {i} exited from frame from right at frame number: {frame_count} ", )
                fish_exited[i] = 'Exited'
            elif end_point[0] < 0 and fish_exited[i] != 'Exited':
                print(f"Fish {i} exited from frame from left at frame number: {frame_count} ")
                fish_exited[i] = 'Exited'
        
    cv2.imshow("image", new_frame)
    cv2.waitKey(0)
        
        #print(f"The object is moving by {translation_x} units along the x-axis and {translation_y} units along the y-axis")
        
    # draw matches
    img_matches = cv2.drawMatches(old_frame, old_keypoints, new_frame, new_keypoints, good_matches, None,
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matches
    #cv2.imshow('Matches', img_matches)
    #cv2.waitKey(0)  
    
  
    
    # draw keypoints on new frame
    #new_frame_with_keypoints = cv2.drawKeypoints(new_frame, new_keypoints, None, color=(0, 255, 0))
    #cv2.imwrite(f"Frame with keypoints{count}.png", new_frame_with_keypoints)
    #count+=1
    
    # display new frame with keypoints
    #cv2.imshow("Frame with keypoints", new_frame_with_keypoints)
    #cv2.waitKey(1)

    # update old frame and continue to next iteration
    old_frame = new_frame
    frame_count+=1