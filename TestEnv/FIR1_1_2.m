%根据频率选择性数字滤波器的性能指标，计算线性相位FIR数字滤波器的理想单位冲激响应

clear all;
close all;
clc;

N=51;                    % 窗的长度
M=(N-1)/2;                % 群时延
Wc=0.3*pi;                % 截止频率


n=-N:N;
d=Wc/pi*sin(Wc*(n-M+eps))./(Wc*(n-M+eps)); %理想低通滤波器 单位冲激响应
figure(1);
stem(n,d,'.','b');
axis([-N N -0.1 0.35]);
title('ideal lowpass filter impulse responses');
xlabel('n');
ylabel('d[n]');

%单位冲激响应加矩形窗
n=0:N-1;
hR=Wc/pi*sin(Wc*(n-M+eps))./(Wc*(n-M+eps)); %加矩形窗后单位冲激响应
figure(2);
stem(n,hR,'.','b');
title('Rectangular windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');
axis([-N N -0.3 0.3]);
figure(3);
subplot(2,1,1);
[HR,w]=freqz(hR);    %加矩形窗后的低通滤波器频率响应
plot(w,20*log(abs(HR)));
axis([0,pi,-150 20]);
title('Rectangular Magnitude Response N=51');
ylabel('|H(ω)|');
xlabel('ω');
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])
subplot(2,1,2);
plot(w,phase(HR));
ylabel('∠H(ω)');
xlabel('ω');
title('Rectangular Phase Response N=51');
grid on;

pause;

%单位冲激响应加海明窗
n=0:N-1;
hH=Hamming(hR,length(n));  %加海明窗后的单位冲激响应
figure(4)

stem(n,hH,'.','b');
title('Hamming windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');

axis([-N N -0.3 0.3]);
figure(5);
subplot(2,1,1);
[HH,w]=freqz(hH);    %加海明窗后的低通滤波器频率响应
plot(w,20*log(abs(HH)));
title('Hamming Magnitude Response N=51');
ylabel('|H(ω)|');
xlabel('ω');
axis([0,pi,-150 20]);
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])
subplot(2,1,2);
plot(w,phase(HH));
title('Hamming Phase Response N=51');
ylabel('∠H(ω)');
xlabel('ω');


