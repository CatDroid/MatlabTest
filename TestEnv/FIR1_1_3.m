%根据频率选择性数字滤波器的性能指标，计算线性相位FIR数字滤波器的理想单位冲激响应

clear all;
close all;
clc;

N1=521;N2=121;                            % 窗的长度
M1=(N1-1)/2; M2=(N2-1)/2;                % 群时延
Wp=0.3*pi;                               % 通带截止频率
Ws=0.4*pi;                               % 阻带截止频率
Wc=(Wp+Ws)/2;


%单位冲激响应加矩形窗
% N1=51时
figure(1);

n=0:N1-1;
hR=Wc/pi*sin(Wc*(n-M1+eps))./(Wc*(n-M1+eps)); %加矩形窗后单位冲激响应
subplot(2,1,1);
stem(n,hR,'.','b');
title('Rectangular windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');


subplot(2,1,2);
[HR,w]=freqz(hR,1,8192);    %加矩形窗后的低通滤波器频率响应
plot(w/pi,20*log10(abs(HR)));
axis([0,1,-150 20]);
title('Rectangular Magnitude Response N=51');
ylabel('|H(ω)|');
xlabel('ω/π');
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])
Mp=max(abs(HR))-1                                %通带的波动量 按公式10.2.3
Ap=20*log10((1+Mp)/(1-Mp))


figure(2);

n=0:N2-1;
hR=Wc/pi*sin(Wc*(n-M2+eps))./(Wc*(n-M2+eps)); %加矩形窗后单位冲激响应
subplot(2,1,1);
stem(n,hR,'.','b');
title('Rectangular windowed impulse responses N=121');
ylabel('h[n]');
xlabel('n');


subplot(2,1,2);
[HR,w]=freqz(hR,1,8192);    %加矩形窗后的低通滤波器频率响应
plot(w/pi,20*log10(abs(HR)));
axis([0,1,-150 20]);
title('Rectangular Magnitude Response N=121');
ylabel('|H(ω)|');
xlabel('ω/π');
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])

Mp=max(abs(HR))-1                                %通带的波动量 按公式10.2.3
Ap=20*log10((1+Mp)/(1-Mp))
