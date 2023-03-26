import cv2
import numpy as np
from matplotlib import pyplot as plt

pairs = [
    ('res/squirrel1.png', 'res/bunny1.png'),
    ('res/cat.jpg', 'res/dog.jpg'),
    ('res/donkey1.png', 'res/zebra1.png'),
]
pair_idx = 0

ksize1, sigma1, ksize2, sigma2 = 39, 11, 21, 5
d1, d2 = 10, 10


def hybrid_image(img1, img2, d1, d2):
    # convert to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # get gaussian kernel
    g1 = cv2.getGaussianKernel(ksize1, sigma1)
    g2 = cv2.getGaussianKernel(ksize2, sigma2)

    # get gaussian image
    img1_g = cv2.sepFilter2D(img1, -1, g1, g1)
    img2_g = cv2.sepFilter2D(img2, -1, g2, g2)

    # get high pass image
    img1_h = img1 - img1_g
    img2_h = img2 - img2_g

    # get hybrid image
    img_h = d1 * img1_h + d2 * img2_h

    # show image
    cv2.imshow('Hybrid Image', img_h.astype(np.uint8))

if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    hybrid_image(img1, img2, d1 / 100, d2 / 100)

    # # create trackbars for ksize and sigma
    # cv2.namedWindow('Hybrid Image')
    # cv2.createTrackbar('d1', 'Hybrid Image', d1, 100, lambda x: None)
    # cv2.createTrackbar('d2', 'Hybrid Image', d2, 100, lambda x: None)

    # while True:
    #     d1 = cv2.getTrackbarPos('d1', 'Hybrid Image')
    #     d2 = cv2.getTrackbarPos('d2', 'Hybrid Image')

    #     # hybrid_image(img1, img2, d1 / 100, d2 / 100)

    #     key = cv2.waitKey(0)
    #     if key == 27:   # esc
    #         break

    # cv2.destroyAllWindows()    


