clear;
clc;
%1
imgu = rgb2gray(imread('F16.tiff')); % 資料強度範圍0~255 但每個語法設定不太一樣 較難控制參數
u = 128;
sigma = 32;
x = 0:255;
Guassian_pdf = (1/sqrt(2*pi*sigma^2))*exp(-((x - u) / sigma).^2);
Guassian_pdf = Guassian_pdf / sum(Guassian_pdf);
img_eq = histeq(imgu, Guassian_pdf);
figure;
subplot(221); imshow(imgu); title('Original Image');
subplot(222); imhist(imgu); title('Original Histogram');
subplot(223); imshow(img_eq); title('Matched Image');
subplot(224); imhist(img_eq); title('Matched Histogram');

%2
img = im2double(rgb2gray(imread('F16.tiff'))); % 資料強度範圍0~1
laplacian_filter_kernal = [0 -1 0; -1 4 -1; 0 -1 0];
img_laplacian_filtered = imfilter(img,laplacian_filter_kernal);
figure; imshow(img_laplacian_filtered); title('Laplacian filter');

%3
img_g_noisy = imnoise(img,'gaussian',0,0.1); % imnoise(img,'gaussian',u,sigma^2)
img_wiener = wiener2(img_g_noisy,[5 5],0.1); % wiener2(I,[M N],noise power)
img_bilat = imbilatfilt(img_g_noisy,0.5,2); % imbilatfilt(I,degreeOfSmoothing,spatialSigma)
img_nlm = imnlmfilt(img_g_noisy,'ComparisonWindowSize',7); % imnlmfilt(I,Name,Value) % Name:degreeOfSmoothing、SearchWindowSize、ComparisonWindowSize
figure;
subplot(221); imshow(img_g_noisy); title('degraded image(Gaussian noise)');
subplot(222); imshow(img_wiener); title('Wiener Filter');
subplot(223); imshow(img_bilat); title('Bilateral Filter');
subplot(224); imshow(img_nlm); title('Non-Local Means Filter');

%4
img_sp_noisy = imnoise(img,'salt & pepper',0.25);
img_wiener = wiener2(img_sp_noisy,[5 5],0.25);
img_median = medfilt2(img_sp_noisy,[5 5]); % medfilt2(img,[3 3])
figure;
subplot(131); imshow(img_sp_noisy); title('Degraded Image(salt & pepper noise)');
subplot(132); imshow(img_wiener); title('Wiener Filter');
subplot(133); imshow(img_median); title('Median Filter');













