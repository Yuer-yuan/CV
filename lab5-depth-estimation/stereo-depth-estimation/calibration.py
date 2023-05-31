import cv2
import numpy as np


def sift_match(img1, img2):
    ratio = 0.65
    sift = cv2.SIFT_create()
    kpts1, desc1 = sift.detectAndCompute(img1, None)
    kpts2, desc2 = sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    pts1 = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    img_match = cv2.drawMatches(img1, kpts1, img2, kpts2, good, None, flags=2)
    cv2.imshow('img_match', img_match)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pts1, pts2


def match_kpts(img1, img2, method='sift'):
    '''
    match keypoints between two images and return the keypoints and matched pairs
    '''
    if method == 'sift':
        return sift_match(img1, img2)
    elif method == 'orb':
        orb = cv2.ORB_create()
        kpts1, desc1 = orb.detectAndCompute(img1, None)
        kpts2, desc2 = orb.detectAndCompute(img2, None)

    else:
        print('Error: method not supported')
        exit()


def get_fundamental(pts1, pts2):
    '''
    get fundamental matrix from matched points
    '''
    fundamental, mask = cv2.findFundamentalMat(pts1, pts2, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.9999, maxIters=1000)
    inliers = mask.ravel().tolist()
    pts1_in = np.array([pts1[i] for i in range(len(pts1)) if inliers[i] == 1])
    pts2_in = np.array([pts2[i] for i in range(len(pts2)) if inliers[i] == 1])
    return fundamental, pts1_in, pts2_in


def get_essential(fundamental, camera_params):
    '''
    get essential matrix from fundamental matrix and camera parameters
    '''
    essential = np.matmul(np.matmul(camera_params['K2'].T, fundamental), camera_params['K1'])
    return essential


def get_rotation_translation(essential, pts):
    '''
    get rotation and translation matrix from essential matrix
    '''
    u, s, vh = np.linalg.svd(essential)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    t1 = u[:, 2]
    t2 = -u[:, 2]
    r1 = np.matmul(np.matmul(u, w), vh)
    r2 = np.matmul(np.matmul(u, w.T), vh)
    poses = [[r1, t1], [r1, t2], [r2, t1], [r2, t2]]

    max_len = 0
    for pos in poses:
        front_pts = []
        for pt in pts:
            X = np.array([pt[0], pt[1], 1])
            V = X - pos[1]
            condition = np.dot(pos[0][2], V)
            if condition > 0:
                front_pts.append(pt)
        if len(front_pts) > max_len:
            max_len = len(front_pts)
            best_pos = pos
    return best_pos
