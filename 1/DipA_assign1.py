#!/usr/bin/env python
# coding: utf-8

# DIPA Quiz #01       Submitted to Sir Naeem Akhtar        Suleman Qamar MS-19-IT-508006
# 
# Add some random noise to your image and then try mean, median, and Gaussian filters to remove it.
# Display original and the resultant three images in a single window

# In[104]:


from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage.util import random_noise
# using this to make pictures appear inline
get_ipython().run_line_magic('matplotlib', 'inline')
img = cv2.imread("selfie.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #we can use grayscale or color image
RGB_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # CV2 default is BGR so changing it to RGB 
HSV_im = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # another color space
plt.imshow(RGB_im)
#plt.imshow(HSV_im)
#plt.imshow(gray)
plt.title('Selfie')
plt.show()


# In[121]:


# Generating Random Noise and adding it to my selfie image (HSV)
img1 = HSV_im.copy() 
cv2.randn(img1,(0,0,0),(255,255,255))
ni = HSV_im + img1 
RGB_ni = cv2.cvtColor(ni, cv2.COLOR_HSV2RGB)
plt.imshow((RGB_ni * 255).astype(np.uint8))
plt.show()


# In[53]:


#Generating noise using random noise function with salt and pepper mode (RGB)
noise_img = random_noise(RGB_im, mode='s&p',amount=0.3)
plt.imshow((noise_img * 255).astype(np.uint8))
plt.show()


# In[98]:


#using noisy image generated with help of random function
mean = cv2.blur(noise_img,(5,5))
gaussian = cv2.GaussianBlur(noise_img,(5,5),0)
noise_img = noise_img.astype('float32')
median = cv2.medianBlur(noise_img,5)
plt.subplot(151),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow(noise_img),plt.title('Noisy')
plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow(mean),plt.title('Mean')
plt.xticks([]), plt.yticks([])
plt.subplot(154),plt.imshow(gaussian),plt.title('Gaussian')
plt.xticks([]), plt.yticks([])
plt.subplot(155),plt.imshow(median),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()


# In[122]:


#using noisy image generated using random numbers and adding them to picture
mean = cv2.blur(ni,(5,5))
gaussian = cv2.GaussianBlur(ni,(5,5),0)
plt.subplot(151),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(152),plt.imshow((ni * 255).astype(np.uint8)),plt.title('Noisy')
plt.xticks([]), plt.yticks([])
plt.subplot(153),plt.imshow((mean * 255).astype(np.uint8)),plt.title('Mean')
plt.xticks([]), plt.yticks([])
plt.subplot(154),plt.imshow((gaussian * 255).astype(np.uint8)),plt.title('Gaussian')
plt.xticks([]), plt.yticks([])
ni = ni.astype('float32')
median = cv2.medianBlur(ni,5)

plt.subplot(155),plt.imshow(median),plt.title('Median')
plt.xticks([]), plt.yticks([])
plt.show()




