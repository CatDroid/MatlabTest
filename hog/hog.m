

clear all; close all; clc;

img=double(imread('lena.jpg'));
imshow(img,[]);
[m n]=size(img);
img=sqrt(img);      %٤��У��

%���������Ե    
fy=[-1 0 1];        %������ֱģ��
fx=fy';             %����ˮƽģ��
Iy=imfilter(img,  fy,  'replicate');     %��ֱ��Ե   imfilter����������������άͼ������˲�
Ix=imfilter(img,  fx,  'replicate');     %ˮƽ��Ե
% ---- �������ÿ��Ԫ�ص��ݶ� 
Ied=sqrt(Ix.^2+Iy.^2);                   %��Եǿ��
% ---- �������ÿ��Ԫ�ص��ݶȴ�С
Iphase=Iy./Ix;                           %��Եб�ʣ���ЩΪinf,-inf,nan������nan��Ҫ�ٴ���һ��
% ---- �������ÿ��Ԫ�ص��ݶȷ���б��(tanֵ) ��Ҫ����������Ƕ� atan


% g = imfilter(f, w, filtering_mode, boundary_options, size_options)
% f Ϊ����ͼ��wΪ�˲���ģ��gΪ�˲���ͼ��
%
% filtering_mode����ָ�����˲���������ʹ�á���ء����ǡ������ : ��corr��( Ĭ��)��������   'conv' ���
% boundary_options���ڴ���߽��������    ��X��  ����ͼ��ı߽�ͨ����ֵX�������ţ��������չ��Ĭ��ֵΪ0
%                                        ��replicate��  ͼ���Сͨ��������߽��ֵ����չ ( �߽����Ԫ�� ���߽� һ��)
%                                        ��symmetric��  ͼ���Сͨ����������߽�����չ
%                                        ��circular��    ͼ���Сͨ����ͼ�񿴳���һ����ά���ں�����һ����������չ
% size_options  ���ͼ��Ĵ�С             ��full��        ���ͼ��Ĵ�С�뱻��չͼ��Ĵ�С��ͬ  
%                                         ��same��(Ĭ��)  ���ͼ��Ĵ�С������ͼ��Ĵ�С��ͬ  ���ͨ�����˲���ģ�����ĵ��ƫ�����Ƶ�ԭͼ���а����ĵ���ʵ��

fid = fopen('exp.txt', 'wt');%��Ҫ�ȴ�һ���ļ�



%��������cell
step=16;                %step*step��������Ϊһ����Ԫ
orient=9;               %����ֱ��ͼ�ķ������
jiao=360/orient;        %ÿ����������ĽǶ���
Cell=cell(1,1);         %���еĽǶ�ֱ��ͼ,cell�ǿ��Զ�̬���ӵģ�����������һ��
ii=1;                      
jj=1;
for i=1:step:m          %��������m/step���������������i=1:step:m-step
    ii=1;
    for j=1:step:n      %ÿ16 * 16������ ��Ϊһ��cell  i j ��Ϊ ͼ��Ԫ������
        tmpx  =  Ix(  i:i+step-1,  j:j+step-1 ); % �� x�����ݶ�ͼ ��  ȡ�� [i,j] ~ [i+16,j+16] ��Χ (������������Ϻ����½�)
        tmped = Ied(  i:i+step-1,  j:j+step-1 ); % temped��Mat 16*16 double����
        tmped=  tmped / sum( sum(tmped) );        % �ֲ���Եǿ�ȹ�һ��
                                                  % Cell��ÿ��Ԫ���ݶȴ�С(��)��һ��
        tmpphase=Iphase(i:i+step-1,j:j+step-1);
        
        Hist=zeros(1,orient);               % ��ǰstep*step���ؿ�ͳ�ƽǶ�ֱ��ͼ,����cell
                                            % B=zeros(m,n)������m��nȫ���� 
        for p=1:step
            for q=1:step  % ���� Cell�е�9x9��Ԫ�� ͳ��
                
                if isnan(tmpphase(p,q))==1  %0/0��õ�nan�����������nan������Ϊ0
                    tmpphase(p,q)=0;
                end
                
                ang=atan(tmpphase(p,q));    %atan�����[-90 90]��֮��
                orgin_angle = ang ;
                angle = ang ;
                
                ang=mod(ang*180/pi,360);    %ȫ��������-90��270
                if tmpx(p,q)<0              %����x����ȷ�������ĽǶ�
                    if ang<90               %����ǵ�һ����
                        ang=ang+180;        %�Ƶ���������
                    end
                    if ang>270              %����ǵ�������
                        ang=ang-180;        %�Ƶ��ڶ�����
                    end
                end
                  
                if tmpx(p,q) < 0   
                    angle = angle + 180 ;
                end
                if ( angle < 0.0 )
                    angle = angle + 360 ;
                end
                fprintf(fid, 'x %f phase %f ori %f angle %f ang %f\n', tmpx(p,q) , tmpphase(p,q) , orgin_angle , angle , ang ); 
                
                
                % ����ĽǶ� �� 0 ~ 360 ��
                
                ang=ang+0.0000001;          %��ֹangΪ0  Matlab�����������1��ʼ ����Hist(0)�ĳ���
                Hist( ceil(ang/jiao) )  =  Hist( ceil(ang/jiao) ) +  tmped(p,q);   %ceil����ȡ����ʹ�ñ�Եǿ�ȼ�Ȩ
            end  % ÿ��Cell��Hist ����Ԫ��(9��) ����������1
        end
        %Hist=Hist/sum(Hist);    %����ֱ��ͼ��һ������һ������û�У���Ϊ�����block�Ժ��ٽ��й�һ���Ϳ���
        Cell{ii,jj}=Hist;       %����Cell��   Cell�Ĵ�С�� 512/16 , 512/16  ���ÿ��Cell��ͳ�ƽ��(���� Ԫ�ظ����� 9 ��  (360/9) )
        ii=ii+1;                %���Cell��y����ѭ������
    end
    jj=jj+1;                    %���Cell��x����ѭ������
end

%��������feature,2*2��cell�ϳ�һ��block,û����ʽ����block
[m n]=size(Cell);
feature=cell(1,(m-1)*(n-1));
for i=1:m-1
   for j=1:n-1           
        f=[];
        f=[f Cell{i,j}(:)' Cell{i,j+1}(:)' Cell{i+1,j}(:)' Cell{i+1,j+1}(:)'];
	f=f./sum(f);%��һ��
        feature{(i-1)*(n-1)+j}=f;
   end
end

fclose(fid)

%���˽�����feature��Ϊ����
%������Ϊ����ʾ��д��
l=length(feature);
f=[];
for i=1:l
    f=[f;feature{i}(:)'];  
end 
figure
mesh(f)