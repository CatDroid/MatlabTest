%这是对 DTFT 的 检测文件

clear all
close all
clc

f0=50; L=200; fs=1000; w0=2*pi*f0/fs;
N=3000;

figure(1);

n=0:L-1;        
xL=cos(2*pi*f0*n/fs);    
axis([0,L-1,-2,2])
hold on
stem(n,xL,'Marker','none') 
plot(n,xL,'r')
xlabel('time samples n')
title('Rectangular Windowed, L=200')  


figure(2);
[DTFT,w]=dtft(xL,N);  
plot(w/pi,abs(DTFT),'b')
axis([0 0.2 0 100]);
xlabel('w in units pi')
title('Magnitude Spectra, L=200')







