function [hog_Feature] = hog(detectedImg, options)  
% -------------------------------------------------------------------------  
% 实现hog（Histogram of gradient）特征的提取过程   
%  
% detectedImg-- 检测窗口包含的图像（灰度图）         %！！！！外部应该先把图像按照检测窗口滑动把窗口的数据作为参数 detectedImg
%                                                  %！！！！最后自己把所有窗口的特征向量 串联起来
% options-- 参数结构体，包含许多参数设置：  
%            cellH, cellW：单元大小    
%            blockH, blockW：块大小  
%            winH, winW：检测窗口大小      
%            stride：块移动步长  
%            bins：直方图长度      
%            flag：梯度方向映射区间（0:[0,pi]  0~180 ,1:[0,2*pi] 0~360 ）  
%            epsilon: 用于做归一化的常量因子  
% @hog_Feature-- 检测窗口图像对应的HOG特征向量，大小为1*M，其中  
%    M = ((winH-blockH)/stride+1)*((winW-blockW)/stride+1)...  
%        *(blockW/cellW)*(blockH/cellH) * bins  
%  
% HOG特征提取步骤：  
% ----------------  
% step 1.由于Dalal在论文中提到色彩和伽马归一化步骤对最终的结果没有影响，故省略该步骤；  
% 利用[-1,0,1]和[1,0,-1]'分别计算图像的x方向和y方向的梯度(这里不采用Sobel或是其他  
% 边缘算子来计算梯度，是因为它们对图像做了平滑处理后再求梯度，这样会丢失很多梯度信息）  
% 然后计算每个像素点对应的梯度幅值和方向：  
%    ||grad|| = |grad_x| + |grad_y|（或||grad|| = sqrt(grad_x^2+grad_y^2)）  
%    gradOri = arctan(grad_y/grad_x) （gradOri属于(-pi/2,pi/2)）  
% 在根据参数flag将每个像素点的梯度方向映射到对应区间中,如果flag为0则选择区间[0,pi]  
% 位于(i,j)位置像素点的方向为:   
%       gradOri(i,j)=gradOri(i,j)<0?gradOri(i,j)+pi, gradOri(i,j)；  
% 如果flag为1选择区间为[0,2*pi],这时需要根据grad_x和grad_y的正负来判断：  
%  (1)grad_x>=0&&grad_y>=0(第一象限) gradOri(i,j)=arctan(grad_y/grad_x)；  
%  (2)grad_x<0&&grad_y>=0(第二象限)  gradOri(i,j)=arctan(grad_y/grad_x)+pi；  
%  (3)grad_x<0&&grad_y<0(第三象限)   gradOri(i,j)=arctan(grad_y/grad_x)+pi；   !!!  (2)(3) 可以合并成 grad_x < 0 角度都加 +pi 
%  (4)grad_x>=0&&grad_y<0(第四象限)  gradOri(i,j)=arctan(grad_y/grad_x)+2*pi；  
% ------------------  
% step 2.为了便于理解，直接写上四层循环，外面两层循环定位block,里面两层定位cell;  
% 一个block对应(blockH*blockW/(cellH*cellW)*bins的特征向量，每个cell对应1*bins  
% 的直方图，计算block的直方图在函数calHist中完成，这里计算直方图需要注意两点：  

% (1)计算cell直方图时，根据像素点梯度幅值进行权值投影，投影时采用软分配方式，即    !!!  这里有做 权值投影  !!!
% 采用插值的方式进行投影，根据梯度方向距离相邻两个区间的中心点的距离进行插值；  

% (2)Dalal在论文中提到，对于R-HOG而言，处理直方图前，在整个block加上一个高斯窗口  
% 这样可以降低block边界像素点的权重，直方图投票值由原先的幅值变为幅值和高斯乘积；  !!!  这里有做  高斯空域滤波 !!!   

% (3)完成block直方图的计算后，需要在整个block范围内进行直方图归一化操作，归一化  
% 方式有多种，这里默认采用L2-norm(hist=hist/sqrt(hist^2+epsilon^2)).           !!!   这里有做  block 归一化  !!!!

% ------------------  
% step 3.合并检测窗口中的所有block的向量（HOG特征向量）  
%  
% 注：整个过程还涉及到一些细节，
% a. 比如导入图像尺度和设置的检测窗口大小不同时，需要完成尺度缩放                !!! 图像尺度 缩成 检测窗口大小 
% b. 在求图像梯度时，边界问题如何处理，是直接填充0还是复制边界，这里直接填充0  
% c. 最后一点就是在计算block直方图时，没有进行三维插值                          !!!  权值投影 没有做三线性插值 !!! 
%    即每个单元中的像,素点只对该单元有投票的权力，对当前block的其他单元没有影响。  
%  
% Author: L.L.He  
% Time: 6/8/2014  
% -------------------------------------------------------------------------  
tic;  % tic 记录命令执行当前时间  toc 记录完成时间
assert(nargin>=1);  
if ~exist('options', 'var')  
    % 如果参数没有指定，则设置为如下默认值  
    options = struct;  
    options.cellH = 8;   options.cellW = 8;  
    options.blockH = 16; options.blockW = 16;  
    options.winH = 64;   options.winW = 128;  
    options.stride = 8;  options.bins = 9;  
    options.flag = 1;    options.epsilon = 1e-4;  
end  
% 处理输入的待检测图像  
[r, c, d] = size(detectedImg);  
if d ~= 1  
    % 需要转换为灰度图  
    detectedImg = rgb2gray(detectedImg);  
end  
detectedImg = double(detectedImg);  
if r~=options.winH && c~=options.winW  
    % 根据检测窗口的大小对输入图像进行尺度缩放(采用双线性插值)  
    detectedImg = imresize(detectedImg, [options.winH options.winW],...  
                           'bilinear');  
end  

temp = size( detectedImg );
temp 
 

% step 1--采用1-D差分卷积核计算x方向和y方向的梯度（幅值和方向）  
mask = [-1, 0, 1];  
[grad, gradOri] = calGrad(detectedImg, mask, options.flag);  
  
% 根据block的大小计算高斯核  
sigma = min(options.blockH, options.blockW)*0.5;  % 由于模板自变量(x,y)在[blockH,blockW]这样定义σ，保证了模板自变量(x,y)在[-2*σ,2*σ]内部
sigma_2 = sigma.^2;  
[X, Y] = meshgrid(  0:options.blockW-1,  0:options.blockH-1);  
% [X, Y] = meshgrid( MatX , MatY) 
% meshgrid产生的X和Y矩阵行数和列数都是一样   行数等于MatY元素个数  列数等于MatX元素个数 
% X每行数据都是MatX 按列张成的一维行向量  Y中每列数据都是MatY 按列张成的一维列向量
X = X - (options.blockW-1)/2;  % 范围变成了 -(options.blockW-1)/2 ~ +(options.blockW-1)/2
Y = Y - (options.blockH-1)/2;  
gaussWeight = exp(-(X.^2+Y.^2)/(2*sigma_2));  % 这里高斯离散近似值 还没有做归一化
% e^[-(x^2+y^2)/(2*σ)]  二维高斯函数 (1/(2*pi*σ^2)) * e^[-(x^2+y^2)/(2*σ) ] 
% 二维模板大小m*n  模板上元素处的值 = (1/(2*pi*σ^2)) * e^[-((x-m/2)^2+(y-n/2)^2)/(2*σ)] 
% 前面的系数在实际应用中常被忽略，因为是离散取样，不能使取样和为1，最后还要做归一化操作

% 这里 sigma 取 block大小的一半 ?? 
  
% 创建一个三维矩阵存放所有block的直方图  
r_tmp = (options.winH-options.blockH)/options.stride+1;   % 计算block的行数目 
c_tmp = (options.winW-options.blockW)/options.stride+1;   % 计算block的列数目 
b_tmp = options.bins *(options.blockH*options.blockW)/... % 计算每个block的特征向量  block中cell数目*bin数目
        (options.cellH*options.cellW);  
blockHist = zeros(r_tmp, c_tmp, b_tmp);   
  
% step 2--计算检测窗口中每个block的直方图(HOG特征向量)  
for i=1:options.stride:(options.winH-options.blockH+1)  
    for j=1:options.stride:(options.winW-options.blockW+1) %i和j为当前block左上角的像素位置  
        block_grad = grad(i:i+options.blockH-1,j:j+options.blockW-1); % 取出这个block中所有像素(梯度大小)(所有Cell下的所有像素)
        block_gradOri = gradOri(i:i+options.blockH-1,j:j+options.blockW-1);  
        % 计算单个block的直方图（投票值为梯度幅值和高斯权重的乘积），并进行归一化处理   
        block_r = floor(i/options.stride)+1;  
        block_c = floor(j/options.stride)+1;  
        blockHist(block_r,block_c,:) = calHist( ...
                    block_grad.*gaussWeight, ... % 先把整个block像素的梯度大小(权值)做高斯空域滤波
                    block_gradOri, ...
                    options);  
    end  
end  
  
% step 3--将所有block的直方图拼接成一维向量作为检测窗口的HOG特征向量  
hog_Feature = reshape(blockHist, [1 numel(blockHist)]);                        % numel  num of elements 矩阵元素个数 
% 把三维矩阵转成一维的 相当于每个block的特征向量收尾连接成一串 成为一个行向量

toc;   % 配合tic计算程序执行时间 
end  
  
% =========================================================================  
function [grad, gradOri] = calGrad(img, mask, flag)  
% -------------------------------------------------------------------------  
% 利用指定的差分卷积核计算x方向和y方向的梯度(包括幅值和方向)  
% img-- 源图像  
% mask-- 计算x方向梯度的差分卷积核(y方向的卷积核是转置后取反)  
% flag-- 梯度方向隐射区间标识  
% @grad-- 梯度幅值  
% @gradOri-- 梯度方向  
% -------------------------------------------------------------------------  
assert(nargin==3);   % nargin 实参的个数 一定是3 assert matlab的断言
xMask = mask    %   [-1,0,1]  行向量
yMask = -mask'   % 正方向  y 向上  x 向右  [1,0,-1]^T 列向量
grad = zeros(size(img));  
gradOri = zeros(size(img));  
grad_x = imfilter(img, xMask);  
grad_y = imfilter(img, yMask);  
% 计算梯度幅值和方向角  
grad = sqrt(double(grad_x.^2 + grad_y.^2));  
if flag == 0  
    % 将梯度方向映射到区间[0,pi]  
    gradOri = atan(grad_y./(grad_x+eps));  
    idx = find(gradOri<0);  
    gradOri(idx) = gradOri(idx) + pi;  
else  
    % 将梯度方向映射到区间[0,2*pi]  
    % 第一象限  
    idx_1 = find(grad_x>=0 & grad_y>=0);  
    gradOri(idx_1) = atan(grad_y(idx_1)./(grad_x(idx_1)+eps));  
    % 第二（三）象限  
    idx_2_3 = find(grad_x<0);  
    gradOri(idx_2_3) = atan(grad_y(idx_2_3)./(grad_x(idx_2_3)+eps)) + pi;  
    % 第四象限  
    idx_4 = find(grad_x>=0 & grad_y<0);  
    gradOri(idx_4) = atan(grad_y(idx_4)./(grad_x(idx_4)+eps)) + 2*pi;  
end  
end  
% =========================================================================  
  
% =========================================================================  
function hist = calHist(block_grad, block_gradOri, options)  
% -------------------------------------------------------------------------  
% 计算单个block的直方图(它由多个cell直方图拼接而成)，并归一化处理  
% block_grad-- block区域对应的梯度幅值矩阵  
% block_gradOri-- block区域对应的梯度方向矩阵  
% options-- 参数结构体，可以得到block中有多少个cell  
% -------------------------------------------------------------------------  
bins = options.bins;  
cellH = options.cellH; cellW = options.cellW;  
blockH = options.blockH; blockW = options.blockW;  
assert(mod(blockH,cellH)==0&&mod(blockW,cellW)==0);  % 正常来说 block的尺寸应该是cell的尺寸的整数倍 这样block中就有整数个cell
hist = zeros(blockH/cellH, blockW/cellW, bins);  % block中每个cell的直方图 1xbins  
% 每个bin对应的角度大小（如果bins为9，每个bin为20度）  
if options.flag == 0  
    anglePerBin = pi/bins;   
    correctVar = pi; % 用来修正currOri为负的情况  
else  
    anglePerBin = 2*pi/bins;  
    correctVar = 2*pi;  
end  
halfAngle = anglePerBin/2; % 后面要用到先计算出来  
for i = 1:blockH  
    for j=1:blockW  
        % 计算当前位置(i,j)属于的单元  
        cell_r = floor((i-1)/cellH)+1;  
        cell_c = floor((j-1)/cellW)+1;  
          
        % 计算当前像素点相连的两个bin并投票  
        currOri = block_gradOri(i,j) - halfAngle;  
        % 为了将第一个bin和最后一个bin连接起来，视0度和180度等价  
        if currOri <= 0  
            currOri = currOri + correctVar;  
        end  
        % 计算该像素点梯度方向所属的两个相连bin的下标  
        pre_idxOfbin = floor(currOri/anglePerBin) + 1;  
        pro_idxOfbin = mod(pre_idxOfbin,bins) + 1;  
        % 向相邻的两个bins进行投票(到中心点的距离作为权重)  
        center = (2*pre_idxOfbin-1)*halfAngle;  
        dist_w = (currOri + halfAngle-center)/anglePerBin;  
        % 只对当前cell中的两个bin做 一维线性插值 投影  没有考虑当前元素对其他cell的影响(三线性插值)
        hist(cell_r,cell_c,pre_idxOfbin) = hist(cell_r,cell_c,pre_idxOfbin)...  
                                           + (1-dist_w)*block_grad(i,j);  
        hist(cell_r,cell_c,pro_idxOfbin) = hist(cell_r,cell_c,pro_idxOfbin)...  
                                           + dist_w*block_grad(i,j);  
    end  
end  
% 将每个cell的直方图合并（拼接一维向量）  
hist = reshape(hist, [1 numel(hist)]);  
% 归一化处理（默认选择L2-norm，可以用其他规则替代）              对在block上进行归一化
hist = hist./sqrt(hist*hist'+ options.epsilon.^2);   % hist*hist' 就是所有元素求和
end  
% =========================================================================  




