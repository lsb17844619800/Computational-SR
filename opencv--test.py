import cv2
from cv2 import dnn_superres
import matplotlib.pyplot as plt
import numpy as np
import math

from data_utils import downscale, upscale

def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return (20 * math.log10(PIXEL_MAX / math.sqrt(mse)))
 
# Create an SR object
sr = dnn_superres.DnnSuperResImpl_create()
 
 
# Read image
img_path = 'validdir/inputs/6.jpg'

image = cv2.imread(img_path)
target = cv2.imread(img_path.replace('inputs','targets'))

scale = 4

# Read the desired model
model_path = "models/FSRCNN_x{}.pb".format(scale)
sr.readModel(model_path)
 
 
# Set the desired model and scale to get correct pre- and post-processing
sr.setModel("fsrcnn", scale)
 
# downscale the image
#input_image = downscale(image,scale)
 
# Upscale the image by model
output_img = sr.upsample(image)

# Upscale the image by cv2.resize
output_img_hat = upscale(image,scale)

base = psnr(target,output_img_hat)
fsrcnn = psnr(target,output_img)
 
plt.figure(figsize=(10,8),dpi=100)
plt.subplot(2,2,1)
plt.title('INPUT')
plt.imshow(image)
plt.subplot(2,2,2)
plt.title('INTER_CUBIC PSNR: %.3fdB'%base)
plt.imshow(output_img_hat)
plt.subplot(2,2,3)
plt.title('TARGET')
plt.imshow(target)
plt.subplot(2,2,4)
plt.title('FSRCNN PSNR: %.3fdB'%fsrcnn)
plt.imshow(output_img)
plt.show()

