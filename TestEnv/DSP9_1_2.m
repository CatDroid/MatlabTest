% ���� ���� ���μӴ� �����Ӵ� ��DTFT�ĵ����ļ�

clc;
clear all;
close all;

f0=50; fs=1000; w0=2*pi*f0/fs;
N=3000; 
L=100;

figure(1);

n=0:300;        
x=cos(2*pi*f0*n/fs);    
subplot(3,1,1);
stem(n,x,'Marker','none');
axis([0,length(n),-1.5,1.5]);
title('signal sampled');

xR=Rectangular(x,L);   % xR Ϊ x �Ӿ��δ�
subplot(3,1,2);
stem(xR,'Marker','none');
title('Rectangular Windowed, L=200');
axis([0,length(n),-1.5,1.5]);

xH=Hamming(x,L);       % xH Ϊ x �Ӻ�����
subplot(3,1,3);
stem(xH,'Marker','none');
title('Hamming Windowed, L=200');
axis([0,length(n),-1.5,1.5]);

figure(2);        % ��xH xR ����DTFT
[Rdtft,wR]=dtft(xR,N);
plot(wR/2/pi*fs,abs(Rdtft),'b');
axis([0 100 0 100]);
hold on
[Hdtft,wH]=dtft(xH,N);
plot(wH/2/pi*fs,abs(Hdtft),'r');
axis([0 100 0 100]);
title('DTFT of both signal');
ylabel('Magnitude Spectrum');
xlabel('f / Hz');
legend('Rectangular signal','Hamming signal');




