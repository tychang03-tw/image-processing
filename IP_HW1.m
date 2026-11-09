clear
clc
% 1.-----------------------
img = imread('F16.tiff');
g_img = rgb2gray(img);

figure;
imshow(g_img);
[count,intensities] = imhist(g_img);
M = max(intensities);
m = min(intensities);
text(10,20, sprintf('Max intensity = %d', M), 'Color','k')
text(10,60, sprintf('Min intensity = %d', m), 'Color','k')

figure;
stem(intensities,count./512^2);% histogram(gray_img);
xlabel('intensities');
ylabel('normalized P(i)');
title('Histogram');
grid on
% 2.----------------------
resize1 = imresize(g_img, 0.5 , 'nearest'); % image,scale,method
rotate1 = imrotate(resize1, 30 , 'nearest'); % image,angle,method
resize2 = imresize(g_img, 0.5 , 'bicubic');
rotate2 = imrotate(resize2, 30, 'bicubic'); % imgout = imwarp(img,tform,method)

figure;
subplot(1,2,1);
imshow(rotate1);
title('Nearest Interpolation , scale = 0.5 , angle = 30(degree)');
subplot(1,2,2);
imshow(rotate2);
title('Bicubic Interpolation , scale = 0.5 , angle = 30(degree)');

