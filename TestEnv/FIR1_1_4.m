%根据频率选择性数字滤波器的性能指标，计算线性相位FIR数字滤波器的理想单位冲激响应

clear all;
close all;
clc;

N1=51;N2=81;                            % 窗的长度
M1=(N1-1)/2; M2=(N2-1)/2;                % 群时延
Wp=0.3*pi;                               % 通带截止频率
Ws=0.4*pi;                               % 阻带截止频率
Wc=(Wp+Ws)/2;


%单位冲激响应加海明窗
n=0:N1-1;
hR=Wc/pi*sin(Wc*(n-M1+eps))./(Wc*(n-M1+eps));
hH=Hamming(hR,length(n));  %加海明窗后的单位冲激响应
figure(1)
subplot(2,1,1);
stem(n,hH,'.','b');
title('Hamming windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');



[HH,w]=freqz(hH,1,4096);    %加海明窗后的低通滤波器频率响应
subplot(2,1,2);
plot(w/pi,20*log10(abs(HH)));
title('Hamming Magnitude Response N=51');
ylabel('|H(ω)|');
xlabel('ω/π');

grid on;
set(gca,'YTickMode','manual','YTick',[-150 -100 -50 -3])

Mp=max(abs(HH))-1                                %通带的波动量 按公式10.2.3
Ap=20*log10((1+Mp)/(1-Mp))


% 时窗长度为 N2=81
n=0:N2-1;
hR=Wc/pi*sin(Wc*(n-M2+eps))./(Wc*(n-M2+eps));
hH=Hamming(hR,length(n));  %加海明窗后的单位冲激响应
figure(2)
subplot(2,1,1);
stem(n,hH,'.','b');
title('Hamming windowed impulse responses N=81');
ylabel('h[n]');
xlabel('n');



[HH,w]=freqz(hH,1,4096);    %加海明窗后的低通滤波器频率响应
subplot(2,1,2);
plot(w/pi,20*log10(abs(HH)));
title('Hamming Magnitude Response N=81');
ylabel('|H(ω)|');
xlabel('ω/π');

grid on;
set(gca,'YTickMode','manual','YTick',[-150 -100 -50 -3])

Mp=max(abs(HH))-1                                %通带的波动量 按公式10.2.3
Ap=20*log10((1+Mp)/(1-Mp))


