clear;
clc;

I = imread('F16.tiff');
Igray = rgb2gray(I);
% 1 --------------------
figure;
% (1-1) Original
subplot(121);
imshow(I);
title('Original');
mask = false(size(Igray));
mask(50:end-50, 50:end-50) = true;
% Active contour
BW = activecontour(Igray, mask, 1000);
subplot(122);
imshow(BW);
title('Active Contour');
% 2 ----------------------
figure;
corners = detectHarrisFeatures(Igray);
% Original
subplot(121);
imshow(I);
title('Original');
% Show corners
subplot(122);
imshow(I); hold on;
plot(corners.selectStrongest(400));
title('Harris Corners');
% 3 -------------------------
figure;
regions = detectMSERFeatures(Igray);
% Original
subplot(131);
imshow(I);
title('Original');
% MSER Regions
subplot(132);
imshow(I); hold on;
plot(regions,'showPixelList',true, 'showEllipses',false);
title('MSER Regions');
subplot(133);
imshow(I); hold on;
plot(regions,'showPixelList',false, 'showEllipses',true);
title('MSER Regions');




