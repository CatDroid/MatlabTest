

clear all; close all; clc;
% img = imread('lena_gray_256.jpg')   % 返回的元素是 uint8 
% subplot(2,1,1); imshow( img,[] );   % imshow 对于元素是double的 要求输入范围要在 0~1  对于元素是uint8的 要求输入在0~255 
% subplot(2,1,2); imshow(img);        % 用了[]就显示min(A)~max(A) 算法实现  b=uint((a-min(a(:)))./(max(a(:))-min(a(:)))*255);
%                                     % 也就是 换了 [] 对比度会提高  并且 转换成了uint 
% return ;
                                    
% rgb_image= imread('color320x240.jpg'); 
% [m n depth]=size(rgb_image); % 如果是RGB图片512*512  [m n]=size(img)返回的是 512,512*3=1536 但是[m n depth]返回的就是[512,512,3]
%                              % [行,列, ] = size()    宽是320 代表列数是320 = n  
% img= rgb2gray(rgb_image); %输入是imread的返回
% img = double( rgb2gray(rgb_image) );
% subplot(2,1,1);  imshow(rgb_image);
% subplot(2,1,2);  imshow(img/256);
% return ;


img = double(  imread('black_write.jpg') );
subplot(2,2,1); imshow(img,[]);  
[m n]=size(img);
img=sqrt(img);                      %伽马校正  x^(1/2)
subplot(2,2,2); imshow(img,[] );    %对比度拓展 
%图片要是灰度图
%return ;
 

m % 512
n % 512

%下面是求边缘    
fy=[-1 0 1];        %定义竖直模板         一维离散微分模板[-1,0,1]
fx=fy';             %定义水平模板
Iy=imfilter(img,  fy,  'replicate');     %竖直边缘   imfilter对任意类型数组或多维图像进行滤波
Ix=imfilter(img,  fx,  'replicate');     %水平边缘
% ---- 计算完成每个元素的梯度 
Ied=sqrt( Ix.^2 + Iy.^2);                   %边缘强度
subplot(2,2,4); imshow(Ied,[]);             %颜色越亮 代表变化越大,越是边缘
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
        % matlab 矩阵索引 (i,j) i 是 行  如果要一行一行遍历，就要每行 中 i固定 修改j
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
                % 在cell中的权值投票 上面直接用了梯度大小 作为权值 
                % 细胞单元（cell）可以是矩形的（rectangular），也可以是星形的（radial）
                % 直方图通道是平均分布在0°-180°（无 向）或0°-360°（有向）范围内
                % 2005 CVPR HOG 作者 发现，采用无向的梯度和9个直方图通道，能在行人检测试验中取得最佳的效果
                % ???????投影  投票在邻域上使用双线性插值（将邻域的bin的中心及其位置来作为插值权重，插入的值为像素的梯度幅值）
                %       加权采用三线性插值 ，即将当前像素的梯度方向大小、像素在cell中的x坐标与y坐标这三个值来 作为插值权重，
                %       而被用来插入的值为像素的梯度幅值
                %       采用三线性插值的好处在于：避免了梯度方向直方图在cell边界和梯度方向量化的bin边界处的突然变化
                %
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
feature=cell(1,(m-1)*(n-1)); % 1x961 cell  @ 512x512图片   961=(32-1)*(32-1) 因为步进是一个cell  block在边界处 
for i=1:m-1
   for j=1:n-1 % block 行和列的步进是 一个cell  
        % 1.组成每个块的特征 
        block=[];  %  Cell{i,j}(:)' 是一个行向量  block是1*36   36 = Block中的Ceil数目(4) * din划分数目(9) 
        block=[  block   Cell{i,j}(:)'    Cell{i,j+1}(:)'   Cell{i+1,j}(:)'    Cell{i+1,j+1}(:)']; % 所有行向量元素合并起来
        block=block./sum(block);  % 归一化  归一化的块描述符就叫作HOG描述子 
        % ??????? 归一化重叠块  归一化的方法 L2-norm L2-Hys L1-norm L1-norm(followed by square root) 
        % 采用L2- Hys L2-norm 和 L1-sqrt方式所取得的效果是一样的 L1-norm稍微表现出一点点不可靠性 但是对于没有被归一化的数据来说，这四种方法都表现出来显着的改进
        
        % 2.加入 整个图像 特征一部分
        %	将从每个 中提取出的“小”HOG特征首尾相连，组合成一个大的一维向量
        feature{  (i-1)*(n-1) + j  }=block;  % 最后得到 1*N 维向量 
   end 
end

% 这个窗口 对应的 一维特征向量 维数n  窗口中的块数 x 块中的胞元数 x 每一个胞元对应的特征向量数 

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

% 将 所有的HOG块描述子组合在一起，形成最终的特征向量，该特征向量就描述了检测窗口的图像内容  也就是上面的feature  
% 将检测窗口中的所有块的HOG描述子组合起来  就形成了最终的特征向量  然后使用 SVM分类器 进行行人检测
% 检测窗口 划分为重叠的块 对这些块计算HOG描述子 形成的特征向量 放到 线性SVM中 进行目标/非目标的二分类
% 检测窗口 在整个图像的所有位置和尺度上 进行扫描


% temp1=[];
% tempp1=[1,2,4,5]; 行向量 
% tempp2=[10,11,13,15];  行向量
% temp1=[temp1 tempp1 tempp2] tempp1 tempp2两个行向量的所有元素 合并
% 打印 temp1 = 1     2     4     5    10    11    13    15 

%   PHOG
%   与 pyramid/金字塔 相结合，即PHOG
%   PHOG指的是，对同一幅图像进行不同尺度的分割，然后计算每个尺度中patch的小HOG，最后将他们连接成一个很长的一维向量
%   对一幅512*512的图像先做3*3的分割，再做6*6的分割，最后做12*12的分割
%   接下来对分割出的patch计算小HOG，假设为12个bin即12维
%   那么就有9*12+36*12+144*12=2268维

%   需要注意的是，在将这些不同尺度上获得的小HOG连接起来时，必须先对其做归一化，
%   因为3*3尺度中的HOG任意一维的数值很可能比12*12尺度中任意一维的数值大很多，这是因为patch的大小不同造成的。
%   PHOG相对于传统HOG的优点，是可以检测到'不同尺度的特征'，表达能力更强。缺点是数据量和计算量都比HOG大了不少




