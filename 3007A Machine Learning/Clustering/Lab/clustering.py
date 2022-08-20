import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

#Reading image
image = plt.imread("peppers.bmp")
#Reshaping the image into (x*y, 3)
imageReshape = image.reshape((image.shape[0]*image.shape[1], image.shape[2]))

print(plt.imshow(image))