import numpy as np
import cv2

img0 = cv2.imread('example2/red.tif')
img1 = cv2.imread('example2/green.tif')
img2 = cv2.imread('example2/blue.tif')
img3 = cv2.imread('example2/infrared.tif')

img = np.array([img0[:,:,0], img1[:,:,0], img2[:,:,0], img3[:,:,0]])

mask0 = cv2.imread('example2/watermask.tif')
mask1 = cv2.imread('example2/urbanmask.tif')
mask2 = cv2.imread('example2/vegtationmask.tif')
mask = np.array([mask0[:,:,0], mask1[:,:,0], mask2[:,:,0]])

n_samples = np.array([[484, 483], [933, 932], [483, 482]])
data_train = np.zeros((n_samples[0,0]+n_samples[1,0]+n_samples[2,0],7))
data_test = np.zeros((n_samples[0,1]+n_samples[1,1]+n_samples[2,1],7))

h, w, _ = img0.shape
n = np.array([0, 0, 0])
n_train = 0
n_test = 0
for y in range(0,h):
    for x in range(0,w):
        for m in range(0,3):
            if (mask[m,y,x] == 255):
                if (n[m] < n_samples[m,0]):
                    data_train[n_train,0:4] = img[:,y,x]
                    data_train[n_train,m+4] = 1.0
                    n_train += 1
                else:
                    data_test[n_test,0:4] = img[:,y,x]
                    data_test[n_test,m+4] = 1.0
                    n_test += 1
                n[m] += 1

with open('example2data.npy', 'wb') as f:
    np.save(f, img)
    np.save(f, data_train)
    np.save(f, data_test)
f.close()