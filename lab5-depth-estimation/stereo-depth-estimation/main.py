import cv2

from data.config import camera_params
from data.config import get_img_path

from calibration import match_kpts
from calibration import get_fundamental
from calibration import get_essential
from calibration import get_rotation_translation

from rectify import rectify
from rectify import draw_epilines

from correspondence import get_disparity

if __name__ == '__main__':
    ''' `datasets` is defined at `data/config.py`
    datasets = [
        'curule',
        'octagon',
        'pendulum',
        'Dataset1',
        'Dataset2',
        'Dataset3',
    ]
    '''
    dataset = 'Dataset1'
    img1_path, img2_path = get_img_path(dataset)
    camera_parameters = camera_params[dataset]
    
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    assert img1 is not None and img2 is not None, 'Error: image read error'
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    pts1, pts2 = match_kpts(img1_gray, img2_gray, method='sift')

    fundamental, pts1_in, pts2_in = get_fundamental(pts1, pts2)

    img1_rf, img2_rf, pt1_rf, pt2_rf = rectify(img1_gray, img2_gray, pts1_in, pts2_in, fundamental)
    
    img1_epilines, img2_epilines = draw_epilines(img1_rf, img2_rf, pt1_rf, pt2_rf, fundamental)

    # img_disparity = get_disparity(img1_rf, img2_rf, 'ssd')
    # cv2.imshow('disparity', img_disparity)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


