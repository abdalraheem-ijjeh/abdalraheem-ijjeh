import math
import random
import numpy as np
import cv2
from KalmanFilter import KalmanFilter

#######################################################################################################
print("Please select a video from 1 to 4: ")
vidNum = int(input())
while vidNum not in [1, 2, 3, 4]:
    print("Please select a video from 1 to 4: ")
    vidNum = int(input())

camera_matrix = np.load('cameraMatrix.npy')
distortion_coefficients = np.load('distortionCoefficients.npy')
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                       distortion_coefficients,
                                                       (640, 480), 1,
                                                       (640, 480))
print(distortion_coefficients)


#######################################################################################################


def merge_obj(img_frame, xx, yy, direct, sf):
    # Update the object's position and scale
    xx += int(direct * 1)  # random.randint(0, 10) / 5)  # Adjust the increment for position
    yy += int(direct * 1)  # random.randint(0, 10) / 5)  # Adjust the increment for position
    sf += direct * scale_increment

    # Check if the object reaches the left or right boundary and change direction
    if xx + object_img.shape[1] <= wrap_around_left:
        sf = 1.0  # Reset the scale when wrapping
        direct = 1  # random.choice([-1, 1])
        new_x = random.randint(10, 300)  # 30, 400
        new_y = random.randint(10, 500)  # 10, 550
        xx, yy = new_x, new_y
    elif xx >= wrap_around_right:
        sf = 1.0
        direct = 1  # random.choice([-1, 1])
        new_x = random.randint(30, 400)  # 30, 400
        new_y = random.randint(10, 550)  # 10, 550
        xx, yy = new_x, new_y

    # Resize the object based on the scale factor
    object_height, object_width, _ = object_img.shape
    new_height = int(object_height * sf)
    new_width = int(object_width * sf)

    if new_height < 1 or new_width < 1:
        return img_frame, int(xx), int(yy), sf, direct

    scaled_object = cv2.resize(object_img, (new_width, new_height))
    ROI = img_frame[int(yy):int(yy) + new_height, int(xx):int(xx) + new_width]

    if ROI.shape != scaled_object.shape:
        return img_frame, xx, yy, sf, direct

    frame_obj = img_frame.copy()
    frame_obj[int(yy):int(yy) + new_height, int(xx):int(xx) + new_width] = scaled_object
    x_static_above = random.randint(50, 51)  # 30, 400
    y_static_above = random.randint(60, 61)  # 10, 550
    frame_obj[y_static_above:y_static_above + 6, x_static_above:x_static_above + 3] = object_img_static
    return frame_obj, int(xx), int(yy), sf, direct


def get_coordinates(img_frame):
    points_coordination_list = []
    for col in range(img_frame.shape[1]):
        column = img_frame[:, col]
        non_zero_indices = np.where(column == 255)[0]
        if len(non_zero_indices) > 0:
            points_coordination_list.append((non_zero_indices[0], col))
    window_size = 51
    points_CoOrds = np.asarray(points_coordination_list)
    points_CoOrds[:, 0] = np.convolve(points_CoOrds[:, 0], (np.ones(window_size) / window_size), mode='same')
    above_horizon = np.zeros_like(img_frame)
    below_horizon = np.zeros_like(img_frame)
    mask = np.ones_like(img_frame)
    for point in points_coordination_list:
        x_inner, y_inner = point
        above_horizon[:x_inner, y_inner] = 255
        below_horizon[x_inner + 15:, y_inner] = 255
        mask[x_inner - 10: x_inner + 10, y_inner] = 0
    return above_horizon, below_horizon, mask


def THRESH(frame_):
    blurred = cv2.GaussianBlur(frame_, (3, 3), 0)
    edge1 = cv2.Canny(blurred, 60, 190)
    edge1 = cv2.dilate(edge1, None, iterations=2)
    edge1 = cv2.Canny(edge1, 60, 190, 1)
    return edge1


def detect_above(frame):
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    img_edges = cv2.Canny(blurred, 50, 190, 3)

    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)

    contours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    min_radius_thresh = 0
    max_radius_thresh = 50

    centers = []
    radius_list = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        # Take only the valid circles
        if (radius >= min_radius_thresh) and (radius < max_radius_thresh):
            centers.append((x, y))
            if radius < 15:
                radius = 15
            radius_list.append(radius)
    return centers, radius_list


def detect_below(frame):
    # kernel = np.ones((3, 3), np.uint8)
    # frame = cv2.dilate(frame, kernel, iterations=5)
    # frame = cv2.erode(frame, kernel, iterations=5)
    cv2.imshow("actual diff", frame)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    img_edges = cv2.Canny(blurred, 25, 75, apertureSize=3)
    ret_, img_thresh = cv2.threshold(img_edges, 10, 255, cv2.THRESH_BINARY)

    cv2.imshow("abs difference", img_thresh)

    contours = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    min_radius_thresh = 1
    max_radius_thresh = 5

    centers = []
    radius_list = []
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)
        # Take only the valid circles
        if (radius >= min_radius_thresh) and (radius < max_radius_thresh):
            centers.append((x, y))
            if radius < 15:
                radius = 15
            radius_list.append(radius)
    return centers, radius_list, img_thresh


def fix_border(frame, angle, scale):
    frame_shape = frame.shape

    matrix = cv2.getRotationMatrix2D(
        (frame_shape[1], frame_shape[0] / 2),
        angle,
        scale
    )

    frame = cv2.warpAffine(frame, matrix, (640, 480), cv2.INTER_LINEAR)
    return frame


def getOptimalFeature(trajectories):
    goodToTrack = False
    centers = []
    total_trajectory_dist_list = []
    if len(trajectories) > 0:
        for i in range(len(trajectories)):
            if len(trajectories[i]) >= 10:
                total_trajectory_dist = math.dist(trajectories[i][0], trajectories[i][-1])
            else:
                continue
            if total_trajectory_dist >= 1:
                total_trajectory_dist_list.append(trajectories[i][-1])
            else:
                continue

        for z in range(len(total_trajectory_dist_list)):
            centers.append(total_trajectory_dist_list[z])
            goodToTrack = True
    return goodToTrack, centers


def get_transformation_mat(set1, set2):
    def rigid_transform_2D(A, B):
        A = A.reshape(-1, 2)
        B = B.reshape(-1, 2)
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        H = np.dot((A - centroid_A).T, (B - centroid_B))
        U, S, Vt = np.linalg.svd(H)
        Rotation_mat = np.dot(Vt.T, U.T)

        if np.linalg.det(Rotation_mat) < 0:
            Vt[1, :] *= -1
            Rotation_mat = np.dot(Vt.T, U.T)

        Translation_mat = -np.dot(Rotation_mat, centroid_A) + centroid_B

        return Rotation_mat, Translation_mat

    # Calculate the rigid transformation matrix

    R, t = rigid_transform_2D(set1, set2)
    print(R)
    print(t)
    euclidean_matrix = np.eye(3)
    euclidean_matrix[:2, :2] = R
    euclidean_matrix[:2, 2] = t
    # print(euclidean_matrix)
    #
    # # Apply the transformation to set2
    # set2_transformed = np.dot(set2, R.T) + t
    #
    # # Calculate the Euclidean distance matrix between the original points in set1 and the transformed points in
    # # set2_transformed
    # euclidean_matrix = np.linalg.norm(set1[:, np.newaxis, :] - set2_transformed, axis=2)
    #
    # print("Euclidean Distance Matrix:")
    # print(euclidean_matrix)
    # # Define source and target points
    # target_points = dst
    # source_points = src
    # # print(source_points.shape, target_points.shape)
    # # Create the transformation equation for a 2D affine transformation
    # A = np.zeros((2 * len(source_points), 6))
    # b = np.zeros((2 * len(source_points)))
    # for i in range(len(source_points)):
    #     A[2 * i][:2] = source_points[i]
    #     A[2 * i][2] = 1
    #     A[2 * i + 1][3:5] = source_points[i]
    #     A[2 * i + 1][5] = 1
    #     b[2 * i:2 * i + 2] = target_points[i]
    # # Solve the system using linear least squares
    # x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=-1)
    # # print(residuals.shape, rank)
    # # Extract the transformation parameters from the solution
    # a, b, tx, c, d, ty = x
    #
    # # Build the final 2D affine transformation matrix
    # matrix = np.array([[a, c, tx],
    #                    [b, d, ty],
    #                    [0, 0, 1]])
    # # matrix = x.reshape((3, 3))
    # # rotation_angle = np.arctan2(matrix[1, 0], matrix[0, 0]) * np.pi / 180
    # # scale = matrix[0, 0] / np.cos(rotation_angle)

    return euclidean_matrix


def FeatureDetection(img_frame):
    blurred = cv2.GaussianBlur(img_frame, (3, 3), 0)
    img_edges = cv2.Canny(blurred, 50, 190, 3)
    ret, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)
    return img_thresh


src_pts_list = []
dst_pts_list = []


def orbMatching(img1, img2):
    MIN_MATCH_COUNT = 4

    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Find the keyPoints and descriptors for both images
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Create a BFMatcher (Brute Force Matcher) with Hamming distance
    bf = cv2.BFMatcher()  # cv2.NORM_HAMMING, crossCheck=True

    # Match descriptors
    matchesORB = bf.match(des1, des2)

    # Sort them in ascending order of distance
    matchesORB = sorted(matchesORB, key=lambda x: x.distance)
    # Filter good matches using a distance threshold
    TM = np.eye(3)
    M = [TM]
    good = []
    for m in matchesORB:
        if m.distance < 1:
            good.append(m)

    if len(good) >= MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts_list.append(src_pts)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts_list.append(dst_pts)
        TM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        M.append(TM)
        matchesMask = mask.ravel().tolist()
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    return M[-1], img3, src_pts_list[-1], dst_pts_list[-1]


def siftMatching(img1, img2):
    MIN_MATCH_COUNT = 10
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    Matches = flann.knnMatch(des1, des2, k=2)
    good = []
    M = np.eye(3)

    for m, n in Matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        src_pts_list.append(src_pts)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts_list.append(dst_pts)
        TMat, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        M = TMat
        matchesMask = mask.ravel().tolist()
        print("Good matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

    return M, img3, src_pts_list[-1], dst_pts_list[-1]


def get_OpticalFLowMotion(img1, img2):
    # prev_points = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
    # curr_points, status, error = cv2.calcOpticalFlowPyrLK(img1, img2, prev_points, None, **lk_params)
    # p0r, status_, error = cv2.calcOpticalFlowPyrLK(img2, img1, curr_points, None, **lk_params)
    # d_ = abs(prev_points - p0r).reshape(-1, 2).max(-1)
    # good = d_ < .01
    # # Select good_ points
    # good_new = curr_points[status == 1]
    # good_old = prev_points[status == 1]

    ################################################################################
    # TM_ = np.eye(3)
    # TM = cv2.findHomography(good_old, good_new, cv2.RANSAC, 1.0)
    # print(TM[0].shape)
    # TM_[:2, :] = TM[0]
    # print(TM)
    #################################################################################
    # Calculating the transformation matrix Hard coding
    #################################################################################
    # TM, angleOfRotation1, scale_ = get_transformation_mat(good_new.astype(np.float32),
    #                                                       good_old.astype(np.float32))

    # print(TM)
    #################################################################################
    # Calculating the transformation matrix using scikit learn package
    #################################################################################
    # TM = ski.transform.estimate_transform('projective', good_old.astype(np.float32),
    #                                       good_new.astype(np.float32))
    # TM = np.asarray(TM)
    # print(TM)
    # angleOfRotation1 = np.arctan2(TM[1, 0], TM[0, 0]) * np.pi / 180
    # scale_ = TM[0, 0] / np.cos(angleOfRotation1)
    #################################################################################
    # Calculating the transformation matrix using Homography
    #################################################################################
    # TM = cv2.findHomography(prev_points, curr_points, method=cv2.RANSAC, ransacReprojThreshold=5)
    # TM = TM[0]
    ################################################################################
    # predicted_frame1 = cv2.warpPerspective(img0,
    #                                        np.linalg.inv(TM),
    #                                        (640, 480),
    #                                        flags=cv2.INTER_LINEAR)
    ############################################################
    # With RANSAC
    # def get_opt_TM():
    #     sift = cv2.ORB_create()
    #
    #     # Detect and compute keypoints and descriptors for both images
    #     keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    #     keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
    #
    #     # Create a BFMatcher (Brute-Force Matcher)
    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     matches = bf.match(descriptors1, descriptors2)
    #
    #     matches = sorted(matches, key=lambda x_: x_.distance)
    #     # Match descriptors using the BFMatcher
    #     print(matches)
    #     # Apply ratio test to select good matches
    #     final_img = cv2.drawMatches(img1, keypoints1,
    #                                 img2, keypoints2, matches[:20], None)
    #     cv2.imshow("Matches", final_img)
    #     good_matches = []
    #     # for m, n in matches:
    #     #     if m.distance < 0.75 * n.distance:
    #     #         good_matches.append(m)
    #
    #     matched_keypoints1 = [keypoints1[match.queryIdx] for match in good_matches]
    #     matched_keypoints2 = [keypoints2[match.trainIdx] for match in good_matches]
    #
    #     # Apply RANSAC to find the best matches
    #     src_pts = np.float32([kp1.pt for kp1 in matched_keypoints1]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kp2.pt for kp2 in matched_keypoints2]).reshape(-1, 1, 2)
    #
    #     H, mask = cv2.findHomography(matched_keypoints1, matched_keypoints2, cv2.RANSAC, 5.0)
    #
    #     return src_pts, dst_pts, H

    #######################################################################
    Matrix, matches, src, dst = siftMatching(img1, img2)
    # Matrix, matches, src, dst = orbMatching(img1, img2)
    #######################################################################
    # TM = get_transformation_mat(src.astype(np.float32),
    #                             dst.astype(np.float32))
    TM, mask = cv2.estimateAffine2D(src.astype(np.float32),
                                    dst.astype(np.float32))
    print("Euclidean\n", TM)
    print("Homo\n", Matrix)
    #######################################################################
    angle = np.arctan2(TM[0, 1], TM[0, 0])  # * np.pi / 180
    scale = TM[0, 0] / np.cos(angle)  # Or scale = 1
    #######################################################################
    cv2.imshow("Matches", matches)
    ##################################################################################
    # Find transformation matrix
    # m = cv2.estimateRigidTransform(src, dst, fullAffine=False)  # will only work with OpenCV-3 or less
    #
    # # Extract traslation
    # dx = m[0, 2]
    # dy = m[1, 2]
    #
    # # Extract rotation angle
    # da = np.arctan2(m[1, 0], m[0, 0])
    ##################################################################################
    predicted_frame_next = cv2.warpAffine(img1, TM, (640, 480),
                                          flags=cv2.INTER_LINEAR)
    predicted_frame_next = fix_border(predicted_frame_next, angle, scale)
    img2 = fix_border(img2, angle, scale)
    ###################################################################################
    Subtracted_prediction_true = cv2.absdiff(predicted_frame_next, img2).astype(np.uint8)
    Subtracted_prediction_true[(predicted_frame_next == 0) | (img2 == 0)] = 0
    #################################################################################
    new_trajectories = []
    cv2.imshow("predicted transformation", predicted_frame_next)
    cv2.imshow("original frame", img2)
    return new_trajectories, Subtracted_prediction_true, matches, predicted_frame_next
    ##########################################################################


def tack_function(path):
    starting_frame = 30
    trajectories_below = []
    x_obj, y_obj = random.randint(80, 250), random.randint(30, 200)
    scale_factor = 0.8
    direction = 1  # random.choice([-1, 1])
    ######################################
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, starting_frame)

    suc, prev_frame = cap.read()
    #####################################
    # prev_frame = prev_frame[y_f:y_f + height_f, x_f:x_f + width_f]
    #####################################
    # prev_frame = resize_frame(prev_frame)
    #####################################
    # Undistorted
    prev_frame = cv2.undistort(prev_frame,
                               cameraMatrix=camera_matrix,
                               distCoeffs=distortion_coefficients,
                               newCameraMatrix=new_camera_matrix)
    #####################################
    prev_frame_gray = prev_frame.copy()

    prev_frame_gray = cv2.cvtColor(prev_frame_gray, cv2.COLOR_BGR2GRAY)
    frame_thresh = THRESH(prev_frame_gray)
    above_prev, below_prev, mask__ = get_coordinates(frame_thresh)
    prev_frame_gray_below = prev_frame_gray  # (below_prev * prev_frame_gray)
    #######################################
    # out1 = cv2.VideoWriter('Videos/Output/Undistorted_video_Tracking_objects%d.mp4' % vidNum,
    #                        cv2.VideoWriter_fourcc(*'MJPG'),
    #                        24, (640, 480))
    # out2 = cv2.VideoWriter('Videos/Output/source_frame%d.mp4' % vidNum,
    #                        cv2.VideoWriter_fourcc(*'DIVX'),
    #                        24, (640 * 1, 480), 0)
    # out3 = cv2.VideoWriter('Videos/Output/destination_frame%d.mp4' % vidNum,
    #                        cv2.VideoWriter_fourcc(*'DIVX'),
    #                        24, (640 * 1, 480), 0)
    # out4 = cv2.VideoWriter('Videos/Output/abs_difference_threshold%d.mp4' % vidNum,
    #                        cv2.VideoWriter_fourcc(*'DIVX'),
    #                        24, (640 * 1, 480), 0)
    # out5 = cv2.VideoWriter('Videos/Output/good_feature_matches%d.mp4' % vidNum,
    #                        cv2.VideoWriter_fourcc(*'MJPG'),
    #                        24, (640 * 2, 480))
    # out6 = cv2.VideoWriter('Videos/Output/abs_difference%d.mp4' % vidNum,
    #                        cv2.VideoWriter_fourcc(*'DIVX'),
    #                        24, (640 * 1, 480), 0)
    #######################################
    diff_image = None
    GOOD_NEW = None
    GOOD_OLD = None
    while True:
        ################################################
        ################################################
        suc, frame = cap.read()
        #######################################################
        # frame = frame[y_f:y_f + height_f, x_f:x_f + width_f]
        #######################################################
        # frame = resize_frame(frame)
        #######################################################
        frame = cv2.undistort(frame,
                              cameraMatrix=camera_matrix,
                              distCoeffs=distortion_coefficients,
                              newCameraMatrix=new_camera_matrix)
        ########################################################
        curr_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_thresh = THRESH(curr_frame_gray)
        above, below, mask_ = get_coordinates(frame_thresh)
        frame_inserted, x_obj, y_obj, direction, scale_factor = merge_obj(frame, x_obj, y_obj, direction,
                                                                          scale_factor)
        frame_inserted_gray = cv2.cvtColor(frame_inserted, cv2.COLOR_BGR2GRAY)
        curr_frame_gray_above = above * frame_inserted_gray
        curr_frame_gray_below = frame_inserted_gray  # cv2.bitwise_and(below, frame_inserted_gray).astype(np.uint8)
        # out2.write(frame_inserted_gray)  # source_frame before applying transformation
        #####################################################################
        # Tracking in the Above
        ####################################################################
        # centers_above, RadiusList = detect_above(curr_frame_gray_above)
        # Kalman_list_above = []
        # if len(centers_above) > 0:
        #     for i in range(len(centers_above)):
        #         Kalman_list_above.append(KalmanFilter(dt=1 / 24,
        #                                               u_x=0.1, u_y=0.1,
        #                                               std_acc=1, x_std_meas=1, y_std_meas=1))
        #     for j, center in enumerate(centers_above):
        #         Kalman_list_above[j].predict()
        #         (x1_above, y1_above, x1_v_above, y1_v_above) = Kalman_list_above[j].update(center)
        #         cv2.rectangle(frame_inserted,
        #                       (int(x1_above[0, 0] - RadiusList[j]), int(y1_above[0, 1] - RadiusList[j])),
        #                       (int(x1_above[0, 0] + RadiusList[j]), int(y1_above[0, 1] + RadiusList[j])),
        #                       (255, 0, 0), 2)
        #####################################################################
        # Tracking in the Below
        ####################################################################
        mask = np.ones_like(curr_frame_gray_below)
        mask *= 255
        prev_points = cv2.goodFeaturesToTrack(prev_frame_gray_below, mask=mask, **feature_params)
        if len(trajectories_below) > 0:
            trajectories_below, diff_image, MATCH, dst_img = get_OpticalFLowMotion(prev_frame_gray_below,
                                                                                   curr_frame_gray_below)
            # out3.write(dst_img)  # destination_frame after applying transformation
            # out5.write(MATCH)  # good_feature_matches

        # Detect the good features to track

        if prev_points is not None:
            for x, y in np.float32(prev_points).reshape(-1, 2):
                trajectories_below.append([(x, y)])
            prev_frame_gray_below = curr_frame_gray_below.copy()

        if diff_image is not None:
            # out6.write(diff_image)
            # diff_image = diff_image * mask_

            centres_below, RadiusList, abs_diff = detect_below(diff_image)
            # out4.write(abs_diff)  # abs_difference

            Kalman_list_below = []
            if len(centres_below) > 0:
                for i in range(len(centres_below)):
                    Kalman_list_below.append(KalmanFilter(dt=1 / 24,
                                                          u_x=0.1, u_y=0.1,
                                                          std_acc=1, x_std_meas=1, y_std_meas=1))
                for j, center in enumerate(centres_below):
                    Kalman_list_below[j].predict()
                    (x1_below, y1_below, x1_v_below, y1_v_below) = Kalman_list_below[j].update(center)
                    cv2.rectangle(frame_inserted,
                                  (int(x1_v_below[0, 0] - 15), int(y1_below[0, 1] - 15)),
                                  (int(x1_below[0, 0] + 15), int(y1_below[0, 1] + 15)),
                                  (0, 0, 255), 2)
        ###########################################################################################
        cv2.imshow("Tracked object", frame_inserted)
        ###########################################################################################
        # out1.write(frame_inserted)  # Undistorted_video_Tracking_objects
        ###########################################################################################
        #################
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    # out1.release()
    # out2.release()
    # out3.release()
    # out4.release()
    # out5.release()
    # out6.release()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    x_f = 240
    y_f = 80
    width_f = 900
    height_f = 550
    scale_increment = 0.02  # Adjust the increment for scaling
    wrap_around_left = 10
    wrap_around_right = 640  # Adjust this value based on your video width
    #######################################
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(11, 11),
                     maxLevel=0,
                     criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 10, 0.01))

    feature_params = dict(maxCorners=500,
                          qualityLevel=0.01,
                          minDistance=0,
                          blockSize=7)

    #######################################
    SMOOTHING_RADIUS = 1
    # Load the object you want to move
    object_img = cv2.imread('object_mask.png')
    object_img = cv2.resize(object_img, (1, 1))
    object_img_static = cv2.imread('Hot_air_balloon.svg.png')
    object_img_static = cv2.resize(object_img_static, (3, 6))
    #######################################
    video_path = 'Videos/Input/ForCamCalibration%d.mp4' % vidNum
    tack_function(video_path)
