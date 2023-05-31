import cv2
import numpy as np


def sum_of_squared_diff(img1, img2):
    '''
    sum of squared difference between two images
    '''
    assert img1.shape == img2.shape, 'the same shape is required'
    diff = img1.astype(np.float32) - img2.astype(np.float32)
    return np.sum(diff * diff)


def blk_cmp(y, x, blk_l, img_r, blk_sz, x_sz, y_sz):
    h, w = img_r.shape[:2]
    x_min = max(0, x - x_sz)
    x_max = min(w - blk_sz, x + x_sz)
    y_min = max(0, y - y_sz)
    y_max = min(h - blk_sz, y + y_sz)
    first = True
    ssd_min = None
    idx_min = None
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            blk_r = img_r[y: y + blk_sz, x: x + blk_sz]
            ssd = sum_of_squared_diff(blk_l, blk_r)
            if first:
                ssd_min = ssd
                idx_min = (y, x)
                first = False
            else:
                if ssd < ssd_min:
                    ssd_min = ssd
                    idx_min = (y, x)
    return idx_min


def ssd_corr(img1, img2):
    h, w = img1.shape[:2]
    blk_sz = 15
    x_sz = 50
    y_sz = 1
    disparity = np.zeros((h, w), dtype=np.float32)
    for y in range(blk_sz, h - blk_sz):
        for x in range(blk_sz, w - blk_sz):
            blk_l = img1[y: y + blk_sz, x: x + blk_sz]
            idx = blk_cmp(y, x, blk_l, img2, blk_sz, x_sz, y_sz)
            disparity[y, x] = abs(idx[1] - x)

            print(f'(y, x) = {(y, x)}')
    return disparity


def get_disparity(img1, img2, method='ssd'):
    if method == 'ssd':
        return ssd_corr(img1, img2)
    else:
        print('Error: method not supported')
        exit()