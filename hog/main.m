clear all; close all; clc;


var options ;


rgb_image= imread('color320x240.jpg'); 
[r,c,d ] = size(rgb_image)
options.winH = r; % ͼƬ��С ����Ӧ�ø� ��ⴰ�� ��һ����  ��ⴰ��������ͼ���ϻ��� 
options.winW = c;

options.cellH = 8;   options.cellW = 8;  
options.blockH = 16; options.blockW = 16;  
options.stride = 8;  
options.bins = 9;  
options.flag = 1;  % 1: ����ѡ���� 0 ~ 2*pi
options.epsilon = 1e-4;  
    
feature = hog(rgb_image , options );

