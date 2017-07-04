%初步分析 物理分辨率与计算分辨率的关系
clear all
close all
clc

fs=10000;
f1=2000; f2=2500; f3=3000;

w1=2*pi*f1/fs; w2=2*pi*f2/fs; w3=2*pi*f3/fs;

L1=20; L2=100;L=800;
N1=32; N2=40; N3=64; N4=800;

figure(1);
n=0:L1-1;
x=cos(w1*n)+cos(w2*n)+cos(w3*n);
[X1,W1]=dft(x,N1);  %DFT
[X2,W2]=dft(x,N2);
[X3,W3]=dft(x,N3);
[X4,W4]=dtft(x,N4); %DTFT

subplot(3,1,1)
stem(W1/pi*(fs/2)/1000,abs(X1),'b.'); 
hold on;
plot(W4/pi*(fs/2)/1000,abs(X4),'r:');
xlabel('f in units kHz');
axis([0 fs/1000 0 12]);
legend('DFT, L=20 N=32','DTFT');
ylabel('Magnitude Spectra');

subplot(3,1,2)
stem(W2/pi*(fs/2)/1000,abs(X2),'b.');
hold on;
plot(W4/pi*(fs/2)/1000,abs(X4),'r:');
xlabel('f in units kHz');
axis([0 fs/1000  0 12]);
legend('DFT, L=20 N=40','DTFT');
ylabel('Magnitude Spectra');

subplot(3,1,3)
stem(W3/pi*(fs/2)/1000,abs(X3),'b.');
hold on;
plot(W4/pi*(fs/2)/1000,abs(X4),'r:');
axis([0 fs/1000  0 12])
xlabel('f in units kHz')
legend('DFT, L=20 N=64','DTFT')
ylabel('Magnitude Spectra')



figure(2);
n=0:L2-1;
x=cos(w1*n)+cos(w2*n)+cos(w3*n);
[X1,W1]=dft(x,N1);  %DFT
[X2,W2]=dft(x,N2);
[X3,W3]=dft(x,N3);
[X4,W4]=dtft(x,N4);  %DTFT

subplot(3,1,1)
stem(W1/pi*(fs/2)/1000,abs(X1),'b.'); 
hold on;
plot(W4/pi*(fs/2)/1000,abs(X4),'r:');
xlabel('f in units kHz');
axis([0 fs/1000 0 50]);
legend('DFT, L=100 N=32','DTFT');
ylabel('Magnitude Spectra');

subplot(3,1,2)
stem(W2/pi*(fs/2)/1000,abs(X2),'b.');
hold on;
plot(W4/pi*(fs/2)/1000,abs(X4),'r:');
xlabel('f in units kHz');
axis([0 fs/1000  0 50]);
legend('DFT, L=100 N=40','DTFT');
ylabel('Magnitude Spectra');

subplot(3,1,3)
stem(W3/pi*(fs/2)/1000,abs(X3),'b.');
hold on;
plot(W4/pi*(fs/2)/1000,abs(X4),'r:');
axis([0 fs/1000  0 50])
xlabel('f in units kHz')
legend('DFT, L=100 N=64','DTFT')
ylabel('Magnitude Spectra')


