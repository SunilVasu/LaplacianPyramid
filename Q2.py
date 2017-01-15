#Question (d) Discuss why or why not the MSE may be non-zero
# MSE= 4485257.0 (Attached the screenshot)
#Answer: The MSE is non zero in this solution because there are certain changes that has creeped in while
#the image is upsampled and downsampled. for example if a image of odd size is taken and the upsampling or
#downsampling mat not provide a correct number of rows or cols hence the value of the MSE would be non #zero.
# Ideally the MSE is to be zero, but in real case due to losses while processing the image the MSE would #remian non zero.

#Note: As I was having problems with opncv in ubuntu 16.04 I have substituted the operation using opencv 
#with python functions likes numpy, Image etc. Due to this the image to be processed needs to be a very #small one as its would take more time to process the output.

import cv2
import numpy as np
import cmath
from PIL import Image
from matplotlib import pyplot as plt
import scipy.ndimage as ndimage
from scipy.misc import imresize as sm

#reference to padding http://numpy-discussion.10968.n7.nabble.com/Style-for-pad-implementation-in-pad-namespace-or-functions-under-np-lib-td5203i20.html
def padwithzeros(vector, pad_width, iaxis, kwargs):
     vector[:pad_width[0]] = 0
     vector[-pad_width[1]:] = 0
     return vector	

def MSE(image1,image2):
	(X,Y)=image1.shape
	mse=0.0
	for m in range(X):
		for n in range(Y):
			mse+=(image1[m,n]-image2[m,n])*(image1[m,n]-image2[m,n])			
	return mse


image = cv2.imread('image.jpg',0)
#cv2.imshow('Original Image',image)
(r,c)=image.shape
cv2.imwrite("OriginalImage.jpg",image)
print 'image r,c', r,c
col=np.size(image,0)
row=np.size(image,1)
array = np.lib.pad(image, 1, padwithzeros)
#original Image
f0=image
print 'fo',f0.shape
cv2.imwrite("Image_f0.jpg",f0)
#cv2.imshow('Image f0',f0)
#Gaussian filter defined
gaussian= np.mat([ [np.float(0.077847),np.float(0.123317),np.float(0.077847)], [np.float(0.123317),np.float(0.195346),np.float(0.123317)], [np.float(0.077847),np.float(0.123317),np.float(0.077847)] ])

print gaussian
#Gaussian filter rotated, flipped for convolution
h=np.rot90(gaussian,2)

#Applying convolution
temp=np.zeros_like(array)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array[i + q, j + w]),np.float(h[i, j]))
                temp[x,y] = np.float(temp[x,y]) + np.float(prod)

print temp
#blurred Image
i0=temp
print 'i0',i0.shape
#Refactoring the image to original size
#cv2.imshow('image i0',temp)
i0 = np.delete(i0, (r), axis=0)
i0 = np.delete(i0, (0), axis=0)
i0 = np.delete(i0, (c), axis=1)
i0 = np.delete(i0, (0), axis=1)
cv2.imwrite("Image_i0.jpg",i0)
#h0 calculated
h0=np.subtract(f0,i0)
print 'h0',h0.shape
#cv2.imshow('image h0',h0) 
cv2.imwrite("Image_h0.jpg",h0)

#SubSampling the image
#f1 = cv2.resize(i0, (0,0), fx=0.5, fy=0.5)
f1 = sm(i0, 0.5)
cv2.imwrite("Image_f1.jpg",f1)
(r,c)=f1.shape
print 'f1',r,c
cv2.imshow('image f1',f1) 

#Blurring the image 2nd time
array2 = np.lib.pad(f1, 1, padwithzeros)
print array2.shape
temp2=np.zeros_like(array2)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array2[i + q, j + w]),np.float(h[i, j]))
                temp2[x,y] = np.float(temp2[x,y]) + np.float(prod)
print temp2
#Refactoring the image to original size
#blurred Image
i1=temp2
i1 = np.delete(i1, (r), axis=0)
i1 = np.delete(i1, (0), axis=0)
i1 = np.delete(i1, (c), axis=1)
i1 = np.delete(i1, (0), axis=1)
cv2.imshow('image i1',i1)
cv2.imwrite("Image_i1.jpg",i1)
#h1 calculated
h1=np.subtract(f1,i1)
#cv2.imshow('image h1',h1) 
cv2.imwrite("Image_h1.jpg",h1)
print 'h1', h1.shape

#subsampling the image
#f2 = cv2.resize(i1, (0,0), fx=0.5, fy=0.5)
#f2 = sm(i1, 0.5)
for x in range(r):
    for y in range (c):
	f2[x,y]=i1[x,y+2]

f2 = f2[:(r/2),:c/2]
(r3,c3)=f2.shape
print 'f2',r3,c3
cv2.imwrite("Image_f2.jpg",f2)
cv2.imshow('image f2',f2) 


#3rd
#Blurring the image 2nd time
(r,c)=f2.shape
array3 = np.lib.pad(f2, 1, padwithzeros)
print array.shape
temp3=np.zeros_like(array3)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array3[i + q, j + w]),np.float(h[i, j]))
                temp3[x,y] = np.float(temp3[x,y]) + np.float(prod)
print temp3
#Refactoring the image to original size
#blurred Image
i2=temp3
i2 = np.delete(i2, (r), axis=0)
i2 = np.delete(i2, (0), axis=0)
i2 = np.delete(i2, (c), axis=1)
i2 = np.delete(i2, (0), axis=1)
cv2.imshow('image i2',i2)
cv2.imwrite("Image_i2.jpg",i2)
#h2 calculated
h2=np.subtract(f2,i2)
#cv2.imshow('image h2',h2) 
cv2.imwrite("Image_h2.jpg",h2)
print 'h2', h2.shape

#subsampling the image
#f2 = cv2.resize(i1, (0,0), fx=0.5, fy=0.5)
f3 = sm(i2, 0.5)
(r,c)=f3.shape
print 'f3',r,c
cv2.imwrite("Image_f3.jpg",f3)
cv2.imshow('image f3',f3) 


#4rd
#Blurring the image 2nd time
(r,c)=f3.shape
array3 = np.lib.pad(f3, 1, padwithzeros)
print array.shape
temp3=np.zeros_like(array3)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array3[i + q, j + w]),np.float(h[i, j]))
                temp3[x,y] = np.float(temp3[x,y]) + np.float(prod)
print temp3
#Refactoring the image to original size
#blurred Image
i3=temp3
i3 = np.delete(i3, (r), axis=0)
i3 = np.delete(i3, (0), axis=0)
i3 = np.delete(i3, (c), axis=1)
i3 = np.delete(i3, (0), axis=1)
cv2.imshow('image i3',i3)
cv2.imwrite("Image_i3.jpg",i3)
#h2 calculated
h3=np.subtract(f3,i3)
#cv2.imshow('image h3',h3) 
cv2.imwrite("Image_h3.jpg",h3)
print 'h3', h3.shape

#subsampling the imae
#f2 = cv2.resize(i1, (0,0), fx=0.5, fy=0.5)
f4 = sm(i3, 0.5)
(r,c)=f4.shape
print 'f4',r,c
cv2.imwrite("Image_f4.jpg",f4)
cv2.imshow('image f4',f4)

#4rd
#Blurring the image 2nd time
(r,c)=f4.shape
array3 = np.lib.pad(f4, 1, padwithzeros)
print array.shape
temp3=np.zeros_like(array3)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array3[i + q, j + w]),np.float(h[i, j]))
                temp3[x,y] = np.float(temp3[x,y]) + np.float(prod)
print temp3
#Refactoring the image to original size
#blurred Image
i4=temp3
i4 = np.delete(i4, (r), axis=0)
i4 = np.delete(i4, (0), axis=0)
i4 = np.delete(i4, (c), axis=1)
i4 = np.delete(i4, (0), axis=1)
cv2.imshow('image i3',i4)
cv2.imwrite("Image_i3.jpg",i4)
#h4 calculated
h4=np.subtract(f4,i4)
#cv2.imshow('image h4',h4) 
cv2.imwrite("Image_h4.jpg",h4)
print 'h4', h4.shape

#subsampling the imae
#f2 = cv2.resize(i1, (0,0), fx=0.5, fy=0.5)
f5 = sm(i4, 0.5)
(r,c)=f5.shape
print 'f5',r,c
cv2.imwrite("Image_f5.jpg",f5)
cv2.imshow('image f5',f5)



#######
#Reverse Operation start
#Upsampling the image
ir4 = sm(f5, 2.0)
print 'ir4',ir4.shape
cv2.imwrite("Image_ir4_upsamp.jpg",ir4)
cv2.imshow('Image ir4_upsamp',ir4)
(r,c)=ir4.shape
#Blur applied
array = np.lib.pad(ir4, 1, padwithzeros)
print array.shape
temp=np.zeros_like(array)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array[i + q, j + w]),np.float(h[i, j]))
                temp[x,y] = np.float(temp[x,y]) + np.float(prod)
print temp

#refactor to original Image 
ir4=temp
ir4 = np.delete(ir4, (r), axis=0)
ir4 = np.delete(ir4, (0), axis=0)
ir4 = np.delete(ir4, (c), axis=1)
ir4 = np.delete(ir4, (0), axis=1)
cv2.imwrite("Image_ir1.jpg",ir4)
#Mat addition
fr4=np.add(ir4,h4)
print 'fr4',fr4.shape
cv2.imshow('Image fr4',fr4)
cv2.imwrite("Image_fr4.jpg",fr4)
#######

#######
#Reverse Operation start
#Upsampling the image
ir3 = sm(f4, 2.0)
print 'ir3',ir3.shape
cv2.imwrite("Image_ir3_upsamp.jpg",ir3)
cv2.imshow('Image ir3_upsamp',ir3)
(r,c)=ir3.shape
#Blur applied
array = np.lib.pad(ir3, 1, padwithzeros)
print array.shape
temp=np.zeros_like(array)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array[i + q, j + w]),np.float(h[i, j]))
                temp[x,y] = np.float(temp[x,y]) + np.float(prod)
print temp

#refactor to original Image 
ir3=temp
ir3 = np.delete(ir3, (r), axis=0)
ir3 = np.delete(ir3, (0), axis=0)
ir3 = np.delete(ir3, (c), axis=1)
ir3 = np.delete(ir3, (0), axis=1)
cv2.imwrite("Image_ir3.jpg",ir3)
#Mat addition
fr3=np.add(ir3,h3)
print 'fr3',fr3.shape
cv2.imshow('Image fr3',fr3)
cv2.imwrite("Image_fr3.jpg",fr3)
#######


#######
#Reverse Operation start
#Upsampling the image
ir2 = sm(f3, 2.0)
print 'ir2',ir2.shape
cv2.imwrite("Image_ir2_upsamp.jpg",ir2)
cv2.imshow('Image ir2_upsamp',ir2)
(r,c)=ir2.shape
#Blur applied
array = np.lib.pad(ir2, 1, padwithzeros)
print array.shape
temp=np.zeros_like(array)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array[i + q, j + w]),np.float(h[i, j]))
                temp[x,y] = np.float(temp[x,y]) + np.float(prod)
print temp

#refactor to original Image 
ir2=temp
ir2 = np.delete(ir2, (r), axis=0)
ir2 = np.delete(ir2, (0), axis=0)
ir2 = np.delete(ir2, (c), axis=1)
ir2 = np.delete(ir2, (0), axis=1)
cv2.imwrite("Image_ir2.jpg",ir2)
#Mat addition
fr2=np.add(ir2,h2)
print 'fr2',fr2.shape
cv2.imshow('Image fr2',fr2)
cv2.imwrite("Image_fr2.jpg",fr2)
#######





#Reverse Operation start
#Upsampling the image
ir1 = sm(f2, 2.0)
print 'ir1',ir1.shape
cv2.imwrite("Image_ir1_upsamp.jpg",ir1)
cv2.imshow('Image ir1_upsamp',ir1)
(r,c)=ir1.shape
#Blur applied
array = np.lib.pad(ir1, 1, padwithzeros)
print array.shape
temp=np.zeros_like(array)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array[i + q, j + w]),np.float(h[i, j]))
                temp[x,y] = np.float(temp[x,y]) + np.float(prod)
print temp

#refactor to original Image 
ir1=temp
ir1 = np.delete(ir1, (r), axis=0)
ir1 = np.delete(ir1, (0), axis=0)
ir1 = np.delete(ir1, (c), axis=1)
ir1 = np.delete(ir1, (0), axis=1)
cv2.imwrite("Image_ir1.jpg",ir1)
#Mat addition
fr1=np.add(ir1,h1)
print 'fr1',fr1.shape
cv2.imshow('Image fr1',fr1)
cv2.imwrite("Image_fr1.jpg",fr1)

#Reverse Operation 2nd - upsampling
ir0 = sm(fr1, 2.0)
print 'ir0',ir0.shape
cv2.imwrite("Image_ir0_upsamp.jpg",ir0)
cv2.imshow('Image ir0',ir0)
(r,c)=ir0.shape

#Blurring
array = np.lib.pad(ir0, 1, padwithzeros)
print array.shape
temp=np.zeros_like(array)
for x in range(r):
    for y in range (c):
        for i in range (0,2):
            for j in range (0,2):
                q = x - 1;
                w = y -1;
		prod=np.dot(np.float(array[i + q, j + w]),np.float(h[i, j]))
                temp[x,y] = np.float(temp[x,y]) + np.float(prod)
print temp
#refactor to original Image
ir0=temp
ir0 = np.delete(ir0, (r), axis=0)
ir0 = np.delete(ir0, (0), axis=0)
ir0 = np.delete(ir0, (c), axis=1)
ir0 = np.delete(ir0, (0), axis=1)
cv2.imwrite("Image_ir0.jpg",ir0)

#mat addition
fr0=np.add(ir0,h0)
print 'fr0',fr0.shape
cv2.imwrite("Image_fr0.jpg",fr0)
cv2.imshow('Image fr0',fr0)
cv2.waitKey(0)
cv2.destroyAllWindows()



mse=MSE(image,ir0)
print "MSE=",mse




