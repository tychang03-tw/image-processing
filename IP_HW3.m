% clear
% clc
%1
% img = im2double(rgb2gray(imread('F16.tiff'))); 
% imgD = dct2(img2);
% [m,n] = size(imgD);
% D_hat = zeros(m,n);
% D_hat(1:m/2, 1:n/2) = imgD(1:m/2, 1:n/2);
% img2= idct2(D_hat);
% 
% figure;
% subplot(121); imshow(img2); title('Original Image');
% subplot(122); imshow(img2); title('Reconstructed Image');


clear
clc
%1
img = im2double(imread('F16.tiff'));
[h, w, c] = size(img);
imgD = zeros(h, w, 3);
img2 = zeros(h, w, 3);
for ch = 1:3
    A = img(:,:,ch);
    D = dct2(A);
    D_low = zeros(h, w);
    D_low(1:h/2, 1:w/2) = D(1:h/2, 1:w/2);
    imgD(:,:,ch) = D_low;
    img2(:,:,ch) = idct2(D_low);
end
figure;
subplot(121); imshow(img); title('Original');
subplot(122); imshow(img2); title('Reconstructed');

%2
imgG = rgb2gray(imread('F16.tiff'));
H = zeros(size(imgG));  % Hue
S = zeros(size(imgG));  % Sat
V = im2double(imgG);    % Intensity

mask = (imgG >= 100 & imgG <= 150);
H(mask) = 0;    % 紅色
S(mask) = 0.5;    % 飽和度多少

HSV = cat(3,H,S,V);
RGB = hsv2rgb(HSV); % 只支援double輸入

figure;
subplot(121); imshow(imgG); title('Original');
subplot(122); imshow(RGB); title('Red-Coded (Intensity Level 100–150)');








