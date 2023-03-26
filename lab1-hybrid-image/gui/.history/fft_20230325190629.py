import cv2
import numpy as np
from matplotlib import pyplot as plt

pairs = [
    ('res/squirrel1.png', 'res/bunny1.png'),
    ('res/')
]
img = cv2.imread('res/bunny1.png', 0)
fft = np.fft.fft2(img)
fftshift = np.fft.fftshift(fft)

magnitude_spectrum = 20*np.log(np.abs(fftshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

