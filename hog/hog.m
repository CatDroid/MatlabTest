function [hog_Feature] = HOG(detectedImg, options)  
% -------------------------------------------------------------------------  
% ʵ��HOG��Histogram of gradient����������ȡ����  
%  
% detectedImg-- ��ⴰ�ڰ�����ͼ�񣨻Ҷ�ͼ��  
% options-- �����ṹ�壬�������������ã�  
%            cellH, cellW����Ԫ��С    
%            blockH, blockW�����С  
%            winH, winW����ⴰ�ڴ�С      
%            stride�����ƶ�����  
%            bins��ֱ��ͼ����      
%            flag���ݶȷ���ӳ�����䣨0:[0,pi]  0~180 ,1:[0,2*pi] 0~360 ��  
%            epsilon: ��������һ���ĳ�������  
% @hog_Feature-- ��ⴰ��ͼ���Ӧ��HOG������������СΪ1*M������  
%    M = ((winH-blockH)/stride+1)*((winW-blockW)/stride+1)...  
%        *(blockW/cellW)*(blockH/cellH) * bins  
%  
% HOG������ȡ���裺  
% ----------------  
% step 1.����Dalal���������ᵽɫ�ʺ�٤���һ����������յĽ��û��Ӱ�죬��ʡ�Ըò��裻  
% ����[-1,0,1]��[1,0,-1]'�ֱ����ͼ���x�����y������ݶ�(���ﲻ����Sobel��������  
% ��Ե�����������ݶȣ�����Ϊ���Ƕ�ͼ������ƽ������������ݶȣ������ᶪʧ�ܶ��ݶ���Ϣ��  
% Ȼ�����ÿ�����ص��Ӧ���ݶȷ�ֵ�ͷ���  
%    ||grad|| = |grad_x| + |grad_y|����||grad|| = sqrt(grad_x^2+grad_y^2)��  
%    gradOri = arctan(grad_y/grad_x) ��gradOri����(-pi/2,pi/2)��  
% �ڸ��ݲ���flag��ÿ�����ص���ݶȷ���ӳ�䵽��Ӧ������,���flagΪ0��ѡ������[0,pi]  
% λ��(i,j)λ�����ص�ķ���Ϊ:   
%       gradOri(i,j)=gradOri(i,j)<0?gradOri(i,j)+pi, gradOri(i,j)��  
% ���flagΪ1ѡ������Ϊ[0,2*pi],��ʱ��Ҫ����grad_x��grad_y���������жϣ�  
%  (1)grad_x>=0&&grad_y>=0(��һ����) gradOri(i,j)=arctan(grad_y/grad_x)��  
%  (2)grad_x<0&&grad_y>=0(�ڶ�����)  gradOri(i,j)=arctan(grad_y/grad_x)+pi��  
%  (3)grad_x<0&&grad_y<0(��������)   gradOri(i,j)=arctan(grad_y/grad_x)+pi��  
%  (4)grad_x>=0&&grad_y<0(��������)  gradOri(i,j)=arctan(grad_y/grad_x)+2*pi��  
% ------------------  
% step 2.Ϊ�˱�����⣬ֱ��д���Ĳ�ѭ������������ѭ����λblock,�������㶨λcell;  
% һ��block��Ӧ(blockH*blockW/(cellH*cellW)*bins������������ÿ��cell��Ӧ1*bins  
% ��ֱ��ͼ������block��ֱ��ͼ�ں���calHist����ɣ��������ֱ��ͼ��Ҫע�����㣺  
% (1)����cellֱ��ͼʱ���������ص��ݶȷ�ֵ����ȨֵͶӰ��ͶӰʱ��������䷽ʽ����  
% ���ò�ֵ�ķ�ʽ����ͶӰ�������ݶȷ����������������������ĵ�ľ�����в�ֵ��  
% (2)Dalal���������ᵽ������R-HOG���ԣ�����ֱ��ͼǰ��������block����һ����˹����  
% �������Խ���block�߽����ص��Ȩ�أ�ֱ��ͼͶƱֵ��ԭ�ȵķ�ֵ��Ϊ��ֵ�͸�˹�˻���  
% (3)���blockֱ��ͼ�ļ������Ҫ������block��Χ�ڽ���ֱ��ͼ��һ����������һ��  
% ��ʽ�ж��֣�����Ĭ�ϲ���L2-norm(hist=hist/sqrt(hist^2+epsilon^2)).  
% ------------------  
% step 3.�ϲ���ⴰ���е�����block��������HOG����������  
%  
% ע���������̻��漰��һЩϸ�ڣ����絼��ͼ��߶Ⱥ����õļ�ⴰ�ڴ�С��ͬʱ����Ҫ  
% ��ɳ߶����ţ�����ͼ���ݶ�ʱ���߽�������δ�����ֱ�����0���Ǹ��Ʊ߽磬����  
% ֱ�����0�����һ������ڼ���blockֱ��ͼʱ��û�н�����ά��ֵ����ÿ����Ԫ�е���  
% �ص�ֻ�Ըõ�Ԫ��ͶƱ��Ȩ�����Ե�ǰblock��������Ԫû��Ӱ�졣  
%  
% Author: L.L.He  
% Time: 6/8/2014  
% -------------------------------------------------------------------------  
tic;  
assert(nargin>=1);  
if ~exist('options', 'var')  
    % �������û��ָ����������Ϊ����Ĭ��ֵ  
    options = struct;  
    options.cellH = 8;   options.cellW = 8;  
    options.blockH = 16; options.blockW = 16;  
    options.winH = 64;   options.winW = 128;  
    options.stride = 8;  options.bins = 9;  
    options.flag = 1;    options.epsilon = 1e-4;  
end  
% ��������Ĵ����ͼ��  
[r, c, d] = size(detectedImg);  
if d ~= 1  
    % ��Ҫת��Ϊ�Ҷ�ͼ  
    detectedImg = rgb2gray(detectedImg);  
end  
detectedImg = double(detectedImg);  
if r~=options.winH && c~=options.winW  
    % ���ݼ�ⴰ�ڵĴ�С������ͼ����г߶�����(����˫���Բ�ֵ)  
    detectedImg = imresize(detectedImg, [options.winH options.winW],...  
                           'bilinear');  
end  
  
% step 1--����1-D��־���˼���x�����y������ݶȣ���ֵ�ͷ���  
mask = [-1, 0, 1];  
[grad, gradOri] = calGrad(detectedImg, mask, options.flag);  
  
% ����block�Ĵ�С�����˹��  
sigma = min(options.blockH, options.blockW)*0.5;  
sigma_2 = sigma.^2;  
[X, Y] = meshgrid(0:options.blockW-1,0:options.blockH-1);  
X = X - (options.blockW-1)/2;  
Y = Y - (options.blockH-1)/2;  
gaussWeight = exp(-(X.^2+Y.^2)/(2*sigma_2));  
  
% ����һ����ά����������block��ֱ��ͼ  
r_tmp = (options.winH-options.blockH)/options.stride+1;  
c_tmp = (options.winW-options.blockW)/options.stride+1;  
b_tmp = options.bins *(options.blockH*options.blockW)/...  
        (options.cellH*options.cellW);  
blockHist = zeros(r_tmp, c_tmp, b_tmp);  
  
% step 2--�����ⴰ����ÿ��block��ֱ��ͼ(HOG��������)  
for i=1:options.stride:(options.winH-options.blockH+1)  
    for j=1:options.stride:(options.winW-options.blockW+1)  
        block_grad = grad(i:i+options.blockH-1,j:j+options.blockW-1);  
        block_gradOri = gradOri(i:i+options.blockH-1,j:j+options.blockW-1);  
        % ���㵥��block��ֱ��ͼ��ͶƱֵΪ�ݶȷ�ֵ�͸�˹Ȩ�صĳ˻���������  
        % ���й�һ������  
        block_r = floor(i/options.stride)+1;  
        block_c = floor(j/options.stride)+1;  
        blockHist(block_r,block_c,:) = calHist(block_grad.*gaussWeight, ...  
                           block_gradOri, options);  
    end  
end  
  
% step 3--������block��ֱ��ͼƴ�ӳ�һά������Ϊ��ⴰ�ڵ�HOG��������  
hog_Feature = reshape(blockHist, [1 numel(blockHist)]);  
toc;  
end  
  
% =========================================================================  
function [grad, gradOri] = calGrad(img, mask, flag)  
% -------------------------------------------------------------------------  
% ����ָ���Ĳ�־���˼���x�����y������ݶ�(������ֵ�ͷ���)  
% img-- Դͼ��  
% mask-- ����x�����ݶȵĲ�־����(y����ľ������ת�ú�ȡ��)  
% flag-- �ݶȷ������������ʶ  
% @grad-- �ݶȷ�ֵ  
% @gradOri-- �ݶȷ���  
% -------------------------------------------------------------------------  
assert(nargin==3);  
xMask = mask;  
yMask = -mask';  
grad = zeros(size(img));  
gradOri = zeros(size(img));  
grad_x = imfilter(img, xMask);  
grad_y = imfilter(img, yMask);  
% �����ݶȷ�ֵ�ͷ����  
grad = sqrt(double(grad_x.^2 + grad_y.^2));  
if flag == 0  
    % ���ݶȷ���ӳ�䵽����[0,pi]  
    gradOri = atan(grad_y./(grad_x+eps));  
    idx = find(gradOri<0);  
    gradOri(idx) = gradOri(idx) + pi;  
else  
    % ���ݶȷ���ӳ�䵽����[0,2*pi]  
    % ��һ����  
    idx_1 = find(grad_x>=0 & grad_y>=0);  
    gradOri(idx_1) = atan(grad_y(idx_1)./(grad_x(idx_1)+eps));  
    % �ڶ�����������  
    idx_2_3 = find(grad_x<0);  
    gradOri(idx_2_3) = atan(grad_y(idx_2_3)./(grad_x(idx_2_3)+eps)) + pi;  
    % ��������  
    idx_4 = find(grad_x>=0 & grad_y<0);  
    gradOri(idx_4) = atan(grad_y(idx_4)./(grad_x(idx_4)+eps)) + 2*pi;  
end  
end  
% =========================================================================  
  
% =========================================================================  
function hist = calHist(block_grad, block_gradOri, options)  
% -------------------------------------------------------------------------  
% ���㵥��block��ֱ��ͼ(���ɶ��cellֱ��ͼƴ�Ӷ���)������һ������  
% block_grad-- block�����Ӧ���ݶȷ�ֵ����  
% block_gradOri-- block�����Ӧ���ݶȷ������  
% options-- �����ṹ�壬���Եõ�block���ж��ٸ�cell  
% -------------------------------------------------------------------------  
bins = options.bins;  
cellH = options.cellH; cellW = options.cellW;  
blockH = options.blockH; blockW = options.blockW;  
assert(mod(blockH,cellH)==0&&mod(blockW,cellW)==0);  
hist = zeros(blockH/cellH, blockW/cellW, bins);  
% ÿ��bin��Ӧ�ĽǶȴ�С�����binsΪ9��ÿ��binΪ20�ȣ�  
if options.flag == 0  
    anglePerBin = pi/bins;   
    correctVar = pi; % ��������currOriΪ�������  
else  
    anglePerBin = 2*pi/bins;  
    correctVar = 2*pi;  
end  
halfAngle = anglePerBin/2; % ����Ҫ�õ��ȼ������  
for i = 1:blockH  
    for j=1:blockW  
        % ���㵱ǰλ��(i,j)���ڵĵ�Ԫ  
        cell_r = floor((i-1)/cellH)+1;  
        cell_c = floor((j-1)/cellW)+1;  
          
        % ���㵱ǰ���ص�����������bin��ͶƱ  
        currOri = block_gradOri(i,j) - halfAngle;  
        % Ϊ�˽���һ��bin�����һ��bin������������0�Ⱥ�180�ȵȼ�  
        if currOri <= 0  
            currOri = currOri + correctVar;  
        end  
        % ��������ص��ݶȷ�����������������bin���±�  
        pre_idxOfbin = floor(currOri/anglePerBin) + 1;  
        pro_idxOfbin = mod(pre_idxOfbin,bins) + 1;  
        % �����ڵ�����bins����ͶƱ(�����ĵ�ľ�����ΪȨ��)  
        center = (2*pre_idxOfbin-1)*halfAngle;  
        dist_w = (currOri + halfAngle-center)/anglePerBin;  
        hist(cell_r,cell_c,pre_idxOfbin) = hist(cell_r,cell_c,pre_idxOfbin)...  
                                           + (1-dist_w)*block_grad(i,j);  
        hist(cell_r,cell_c,pro_idxOfbin) = hist(cell_r,cell_c,pro_idxOfbin)...  
                                           + dist_w*block_grad(i,j);  
    end  
end  
% ��ÿ��cell��ֱ��ͼ�ϲ���ƴ��һά������  
hist = reshape(hist, [1 numel(hist)]);  
% ��һ������Ĭ��ѡ��L2-norm���������������������  
hist = hist./sqrt(hist*hist'+ options.epsilon.^2);  
end  
% =========================================================================  