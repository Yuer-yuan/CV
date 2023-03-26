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



if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)
    img2 = cv2.imread(pairs[pair_idx][1], 0)

    # create trackbars for ksize and sigma
    cv2.namedWindow('Hybrid Image')
    cv2.createTrackbar('d1', 'Hybrid Image', d1, 100, lambda x: None)
    cv2.createTrackbar('d2', 'Hybrid Image', d2, 100, lambda x: None)

    while True:
        d1 = cv2.getTrackbarPos('d1', 'Hybrid Image')
        d2 = cv2.getTrackbarPos('d2', 'Hybrid Image')

        # hybrid_image(img1, img2, d1 / 100, d2 / 100)

        key = cv2.waitKey(0)
        if key == 27:   # esc
            break

    cv2.destroyAllWindows()    


