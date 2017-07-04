

clear all; close all; clc;

rgb_image= imread('lena.jpg');
%img=rgb2gray(rgb_image); %输入是imread的返回 
img = double( rgb2gray(rgb_image) );
%imshow(img/256); %imshow 输入范围要在 0 ~1 
%return ;

%[m n depth]=size(img); % 如果是RGB图片512*512  [m n]=size(img)返回的是 512,512*3=1536 但是[m n depth]返回的就是[512,512,3]
[m n]=size(img);
%img=sqrt(img);      %伽马校正
imshow(img/256);
%图片要是灰度图
 

m % 512
n % 512

%下面是求边缘    
fy=[-1 0 1];        %定义竖直模板
fx=fy';             %定义水平模板
Iy=imfilter(img,  fy,  'replicate');     %竖直边缘   imfilter对任意类型数组或多维图像进行滤波
Ix=imfilter(img,  fx,  'replicate');     %水平边缘
% ---- 计算完成每个元素的梯度 
Ied=sqrt(Ix.^2+Iy.^2);                   %边缘强度
% ---- 计算完成每个元素的梯度大小
Iphase=Iy./Ix;                           %边缘斜率，有些为inf,-inf,nan，其中nan需要再处理一下
% ---- 计算完成每个元素的梯度方向斜率(tan值) 还要反过来计算角度 atan


% g = imfilter(f, w, filtering_mode, boundary_options, size_options)
% f 为输入图像，w为滤波掩模，g为滤波后图像。
%
% filtering_mode用于指定在滤波过程中是使用“相关”还是“卷积” : ‘corr’( 默认)相关来完成   'conv' 卷积
% boundary_options用于处理边界充零问题    ‘X’  输入图像的边界通过用值X（无引号）来填充扩展其默认值为0
%                                        ‘replicate’  图像大小通过复制外边界的值来扩展 ( 边界外的元素 跟边界 一样)
%                                        ‘symmetric’  图像大小通过镜像反射其边界来扩展
%                                        ‘circular’    图像大小通过将图像看成是一个二维周期函数的一个周期来扩展
% size_options  输出图像的大小             ‘full’        输出图像的大小与被扩展图像的大小相同  
%                                         ‘same’(默认)  输出图像的大小与输入图像的大小相同  这可通过将滤波掩模的中心点的偏移限制到原图像中包含的点来实现

% fid = fopen('exp.txt', 'wt');%需要先打开一个文件



%下面是求cell
step=16;                %step*step个像素作为一个单元
orient=9;               %方向直方图的方向个数
jiao=360/orient;        %每个方向包含的角度数
Cell=cell(1,1);         %所有的角度直方图,cell是可以动态增加的，所以先设了一个
ii=1;                      
jj=1;

for i=1:step:m          %如果处理的m/step不是整数，最好是i=1:step:m-step
    ii=1;
    for j=1:step:n      %每16 * 16个像素 作为一个cell  i j 作为 图像元素索引
        tmpx  =  Ix(  i:i+step-1,  j:j+step-1 ); % 在 x方向梯度图 中  取出 [i,j] ~ [i+16,j+16] 范围 (矩形区域的左上和右下角)
        tmped = Ied(  i:i+step-1,  j:j+step-1 ); % temped是Mat 16*16 double类型
        tmped=  tmped / sum( sum(tmped) );        % 局部边缘强度归一化
                                                  % Cell中每个元素梯度大小(摸)归一化
        tmpphase=Iphase(i:i+step-1,j:j+step-1);
        
        Hist=zeros(1,orient);               % 当前step*step像素块统计角度直方图,就是cell
                                            % B=zeros(m,n)：生成m×n全零阵
                                            % 1 x orient 行向量 
        for p=1:step
            for q=1:step  % 遍历 Cell中的9x9个元素 然后做统计
                
                if isnan(tmpphase(p,q))==1  %0/0会得到nan，如果像素是nan，重设为0
                    tmpphase(p,q)=0;
                end
                
                ang=atan(tmpphase(p,q));    %atan求的是[-90 90]度之间  返回的是弧度
                
%                 ang=mod(ang*180/pi,360);    %全部变正，-90变270
%                 if tmpx(p,q)<0              %根据x方向确定真正的角度
%                     if ang<90               %如果是第一象限
%                         ang=ang+180;        %移到第三象限
%                     end
%                     if ang>270              %如果是第四象限
%                         ang=ang-180;        %移到第二象限
%                     end
%                 end
                  
                ang = ang * 180 / pi ; % 弧度转角度
                if tmpx(p,q) < 0   
                    ang = ang + 180 ;
                end
                if ( ang < 0.0 )
                    ang = ang + 360 ;
                end
                %fprintf(fid, 'x %f phase %f ori %f angle %f ang %f\n', tmpx(p,q) , tmpphase(p,q) , orgin_angle , angle , ang ); 
                
                % 这里的角度 在 0 ~ 360 度
                ang=ang+0.0000001;          %防止ang为0  Matlab的数组坐标从1开始 避免Hist(0)的出现
                Hist( ceil(ang/jiao) )  =  Hist( ceil(ang/jiao) ) +  tmped(p,q);   %ceil向上取整，使用边缘强度加权
            end  % 每个Cell的Hist 所有元素(9个) 加起来和是1
        end
        %Hist=Hist/sum(Hist);    %方向直方图归一化，这一步可以没有，因为是组成block以后再进行归一化就可以
        Cell{ii,jj}=Hist;       %放入Cell中   Cell的大小是 512/16 , 512/16  存放每个Cell的统计结果(向量 元素个数是 9 个  (360/9) )
        ii=ii+1;                %针对Cell的y坐标循环变量
    end
    jj=jj+1;                    %针对Cell的x坐标循环变量
end

 

%元胞数组  每个元素可以是不同的类型和内容  可动态分配/增大
% a = { 20, 'matlab' ;  ones(2,3) , 1:3 }
% a(1,2)   = 'matlab'
% a{1,2}   = matlab 
%{}  元胞的内容   () 元胞

%下面是求feature,2*2个cell合成一个block,没有显式的求block
[m n]=size(Cell); % 32x32 cell  @ 512x512图片  
feature=cell(1,(m-1)*(n-1)); % 1x961 cell  @ 512x512图片
for i=1:m-1
   for j=1:n-1           
        f=[];
        f=[  f   Cell{i,j}(:)'    Cell{i,j+1}(:)'   Cell{i+1,j}(:)'    Cell{i+1,j+1}(:)'];
        f=f./sum(f);%归一化
        feature{(i-1)*(n-1)+j}=f;
   end
end

% fclose(fid)

%到此结束，feature即为所求
%下面是为了显示而写的
len=length(feature);
f=[];
for i=1:len
    f=[f;feature{i}(:)'];  
end 
figure
mesh(f)