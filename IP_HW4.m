clear;
clc;
%1 ----------------
img = rgb2gray(imread('F16.tiff'));
edge_img = edge(img, 'canny');
figure; imshow([im2double(img) im2double(edge_img)]);
%2 ---------------
edge_img = imclearborder(edge_img);  % 去除貼邊的邊緣
edge_img = bwareaopen(edge_img, 30); % 移除太小的雜點（面積小於30的物件）
[H,theta,rho] = hough(edge_img); % Hough 轉換
peaks = houghpeaks(H,1000,'Threshold', ceil(0.2 * max(H(:)))); 
% 找出Hough累加器中的峰值 
% 閾值太高 → 只會找到最強的直線，可能漏掉一些細線 
% 閾值太低 → 會找到很多噪聲線
lines = houghlines(edge_img,theta,rho,peaks,'FillGap',5,'MinLength',12); %根據峰值偵測直線
% FillGap：兩段直線之間的最大間距，如果小於這個值就會合併成一條線，MinLength：忽略太短的線段
figure; imshow(img);
hold on;
for k = 1:length(lines)
   xy = [lines(k).point1; lines(k).point2];
   plot(xy(:,1), xy(:,2),'Color', 'green');
   plot(xy(1,1), xy(1,2), 'x','Color', 'yellow');
   plot(xy(2,1), xy(2,2), 'x','Color', 'red');
end
hold off;
%3 ----------------
T = graythresh(img);    % 找出Otsu閾值T
img_binary = imbinarize(img,T); % 根據thershold二值化
figure; imshow([im2double(img) img_binary]);
%4 ----------------
L = imsegkmeans(img,2);
L_n = L < (1+2)/2;
figure; imshow([im2double(img) double(L_n)]);









