function filt=MyGaussian(varargin) 
% 高斯函数的离散近似   高斯函数上取离散点的值 并归一化
% varargin{1}  模板尺寸mxn  若是一个数字 那就是m=n 
% varargin{2}  方差 sigma 

%参数初始化，使用varargin处理可变参数情况  
siz=varargin{1};%模板尺寸
if(numel(siz)==1)  
    siz=[siz,siz];  
end  
std=varargin{2};%方差  
centa = (siz(1)+1)/2;%此处不要取整  
centb = (siz(1)+1)/2;  
  
filt = zeros(siz(1),siz(2));  
summ=0;  
for i=1:siz(1)  
    for j=1:siz(2)  
        radius = ((i-centa)^2+(j-centb)^2);  
        filt(i,j) = exp(-(radius/(2*std^2))); % 这里没有除以  2*pi*sigma^2   
        summ=summ+filt(i,j);  % 所有在(连续)高斯函数上取得的离散值的和 
    end  
end  
filt=filt/summ; %归一化  

% 二维模板大小m*n  
% 模板上元素处的值 = (1/(2*pi*σ^2)) * e^[-((x-m/2)^2+(y-n/2)^2)/(2*σ)]  
% 前面的系数在实际应用中常被忽略，因为是离散取样，不能使取样和为1，最后还要做归一化操作
% 即 e^[-((x-m/2)^2+(y-n/2)^2)/(2*σ)] 

% "3σ准则"，即数据分布在[u-3σ, u+3σ ]的概率是0.9974
% u=0时候 就是 [-3σ,3σ]
% [0,3σ+3σ]   模板尺寸就是  0~6σ x 0~6σ 
% σ参数给定 e.g σ = 1  那模板最好是 0~6 x 0~6  e.g 4x4 4x6 
% 由于最后对离散值做了归一化 所以返回的矩阵总和一定是 1 
