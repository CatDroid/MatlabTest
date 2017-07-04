clear all;
close all;
clc;

figure(1);

N=81;
n=0:N-1;
y=Hamming(ones(N),N);
figure(1);
[h,w]=freqz(y);
plot(w,h);
figure(2);
plot(w,phase(h));

figure(1);
[h,w]=freqz(d)

figure(2);
plot(n,d);
axis([0 200 -0.1 0.4]);
figure(3);
[h,w]=freqz(d);
plot(w,abs(h));

figure(4);
h=Hamming(d,length(d));
plot(n,h);
axis([0 200 -0.1 0.4]);
figure(5);
[h,w]=freqz(h);
plot(w,abs(h));

