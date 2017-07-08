clear all; close all; clc;


var options ;


rgb_image= imread('color320x240.jpg'); 
[r,c,d ] = size(rgb_image)
options.winH = r; % 图片大小 正常应该跟 检测窗口 不一样大  检测窗口在整个图像上滑动 
options.winW = c;

options.cellH = 8;   options.cellW = 8;  
options.blockH = 16; options.blockW = 16;  
options.stride = 8;  
options.bins = 9;  
options.flag = 1;  % 1: 区间选择在 0 ~ 2*pi
options.epsilon = 1e-4;  
    
feature = hog(rgb_image , options );

