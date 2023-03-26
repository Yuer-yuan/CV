import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('res.bunny', 0)
fft = np.fft.fft2(img)
fftshift = np.fft.fftshift(fft)
