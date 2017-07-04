%����Ƶ��ѡ���������˲���������ָ�꣬����������λFIR�����˲��������뵥λ�弤��Ӧ

clear all;
close all;
clc;

N=51;                    % ���ĳ���
M=(N-1)/2;                % Ⱥʱ��
Wc=0.3*pi;                % ��ֹƵ��


n=-N:N;
d=Wc/pi*sin(Wc*(n-M+eps))./(Wc*(n-M+eps)); %�����ͨ�˲��� ��λ�弤��Ӧ
figure(1);
stem(n,d,'.','b');
axis([-N N -0.1 0.35]);
title('ideal lowpass filter impulse responses');
xlabel('n');
ylabel('d[n]');

%��λ�弤��Ӧ�Ӿ��δ�
n=0:N-1;
hR=Wc/pi*sin(Wc*(n-M+eps))./(Wc*(n-M+eps)); %�Ӿ��δ���λ�弤��Ӧ
figure(2);
stem(n,hR,'.','b');
title('Rectangular windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');
axis([-N N -0.3 0.3]);
figure(3);
subplot(2,1,1);
[HR,w]=freqz(hR);    %�Ӿ��δ���ĵ�ͨ�˲���Ƶ����Ӧ
plot(w,20*log(abs(HR)));
axis([0,pi,-150 20]);
title('Rectangular Magnitude Response N=51');
ylabel('|H(��)|');
xlabel('��');
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])
subplot(2,1,2);
plot(w,phase(HR));
ylabel('��H(��)');
xlabel('��');
title('Rectangular Phase Response N=51');
grid on;

pause;

%��λ�弤��Ӧ�Ӻ�����
n=0:N-1;
hH=Hamming(hR,length(n));  %�Ӻ�������ĵ�λ�弤��Ӧ
figure(4)

stem(n,hH,'.','b');
title('Hamming windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');

axis([-N N -0.3 0.3]);
figure(5);
subplot(2,1,1);
[HH,w]=freqz(hH);    %�Ӻ�������ĵ�ͨ�˲���Ƶ����Ӧ
plot(w,20*log(abs(HH)));
title('Hamming Magnitude Response N=51');
ylabel('|H(��)|');
xlabel('��');
axis([0,pi,-150 20]);
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])
subplot(2,1,2);
plot(w,phase(HH));
title('Hamming Phase Response N=51');
ylabel('��H(��)');
xlabel('��');


