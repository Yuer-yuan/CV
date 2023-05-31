import cv2
import numpy as np


def rectify(img1, img2, pts1, pts2, fundamental):
    '''
    rectify two images
    '''
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    _, homography1, homography2 = cv2.stereoRectifyUncalibrated(pts1, pts2, fundamental, imgSize=(w1, h1))
    img1_rf = cv2.warpPerspective(img1, homography1, (w1, h1))
    img2_rf = cv2.warpPerspective(img2, homography2, (w2, h2))
    pts1_rf = np.zeros(pts1.shape, dtype=np.float32)
    pts2_rf = np.zeros(pts2.shape, dtype=np.float32)
    for i in range(pts1.shape[0]):
        src1 = np.array([pts1[i, 0], pts1[i, 1], 1], dtype=np.float32)
        dst1 = np.dot(homography1, src1)
        pts1_rf[i, 0] = int(dst1[0] / dst1[2])
        pts1_rf[i, 1] = int(dst1[1] / dst1[2])
        src2 = np.array([pts2[i, 0], pts2[i, 1], 1])
        dst2 = np.dot(homography2, src2)
        pts2_rf[i, 0] = int(dst2[0] / dst2[2])
        pts2_rf[i, 1] = int(dst2[1] / dst2[2])

    # cv2.imshow('img1_rectified', img1_rf)
    # cv2.imshow('img2_rectified', img2_rf)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img1_rf, img2_rf, pts1_rf, pts2_rf


def draw_epilines(img1, img2, pts1, pts2, fundamental):
    '''
    draw epilines on two images
    '''
    w = img1.shape[1]
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental)
    lines1 = lines1.reshape(-1, 3)
    img1_epilines = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    for line in lines1:
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = w, int(-(line[2] + line[0] * w) / line[1])
        img1_epilines = cv2.line(img1_epilines, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for pt in pts1:
        img1_epilines = cv2.circle(img1_epilines, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
    
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental)
    lines2 = lines2.reshape(-1, 3)
    img2_epilines = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for line in lines2:
        x0, y0 = 0, int(-line[2] / line[1])
        x1, y1 = w, int(-(line[2] + line[0] * w) / line[1])
        img2_epilines = cv2.line(img2_epilines, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for pt in pts2:
        img2_epilines = cv2.circle(img2_epilines, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)

    # cv2.imshow('img1_epilines', img1_epilines)
    # cv2.imshow('img2_epilines', img2_epilines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return img1_epilines, img2_epilines
