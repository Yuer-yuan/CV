import cv2
import numpy as np
from matplotlib import pyplot as plt

pairs = [
    ('res/squirrel1.png', 'res/bunny1.png'),
    ('res/cat.jpg', 'res/dog.jpg'),
    ('res/donkey1.png', 'res/zebra1.png'),
]
pair_idx = 0

ksize1 = 

if __name__ == '__main__':
    img1 = cv2.imread(pairs[pair_idx][0], 0)
    img2 = cv2.imread(pairs[pair_idx][1], 0)



img1 = cv2.imread(pairs[0][0], 0)

img = cv2.imread('res/bunny1.png', 0)
fft = np.fft.fft2(img)
fftshift = np.fft.fftshift(fft)

magnitude_spectrum = 20*np.log(np.abs(fftshift))

plt.subplot(121),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()

