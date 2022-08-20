import cv2
import numpy as np
from skimage.io import imread, imsave, imshow, show, imshow_collection, imread_collection
import matplotlib.pyplot as plt

def D(im):
    '''
    Returns array of distances from center of frequency rectangle to point (x, y)
    '''
    u = np.arange(im.shape[0])
    v = np.arange(im.shape[1])

    idx = np.where(u>len(u)/2)
    idy = np.where(v>len(v)/2)
    u[idx] = u[idx]-len(u)
    v[idy] = v[idy]-len(v)

    V,U = np.meshgrid(v, u)
    d = np.sqrt(V**2 + U**2)
    return np.fft.fftshift(d)


def H(im, d0, w):
    n = 2
    d = D(im)
    dw = d*w
    den = d**2 - d0**2
    return 1/(1+(dw/den)**(2*n))


image2 = imread("../images/lavender.png")

dft = cv2.dft(np.float32(image2),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))


rows, cols = image2.shape
crow,ccol = rows//2 , cols//2
mask = H(image2, 10, 50)
mask = np.fft.fftshift(mask)

maskTemp = np.ones((rows, cols, 2))

fshift = dft_shift*maskTemp
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.figure(figsize=(11,6))
plt.subplot(121),plt.imshow(image2, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
plt.title('Low Pass Filter'), plt.xticks([]), plt.yticks([])
plt.show()