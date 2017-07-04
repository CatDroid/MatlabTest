%����Ƶ��ѡ���������˲���������ָ�꣬����������λFIR�����˲��������뵥λ�弤��Ӧ

clear all;
close all;
clc;

N1=51;N2=81;                            % ���ĳ���
M1=(N1-1)/2; M2=(N2-1)/2;                % Ⱥʱ��
Wp=0.3*pi;                               % ͨ����ֹƵ��
Ws=0.4*pi;                               % �����ֹƵ��
Wc=(Wp+Ws)/2;


%��λ�弤��Ӧ�Ӻ�����
n=0:N1-1;
hR=Wc/pi*sin(Wc*(n-M1+eps))./(Wc*(n-M1+eps));
hH=Hamming(hR,length(n));  %�Ӻ�������ĵ�λ�弤��Ӧ
figure(1)
subplot(2,1,1);
stem(n,hH,'.','b');
title('Hamming windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');



[HH,w]=freqz(hH,1,4096);    %�Ӻ�������ĵ�ͨ�˲���Ƶ����Ӧ
subplot(2,1,2);
plot(w/pi,20*log10(abs(HH)));
title('Hamming Magnitude Response N=51');
ylabel('|H(��)|');
xlabel('��/��');

grid on;
set(gca,'YTickMode','manual','YTick',[-150 -100 -50 -3])

Mp=max(abs(HH))-1                                %ͨ���Ĳ����� ����ʽ10.2.3
Ap=20*log10((1+Mp)/(1-Mp))


% ʱ������Ϊ N2=81
n=0:N2-1;
hR=Wc/pi*sin(Wc*(n-M2+eps))./(Wc*(n-M2+eps));
hH=Hamming(hR,length(n));  %�Ӻ�������ĵ�λ�弤��Ӧ
figure(2)
subplot(2,1,1);
stem(n,hH,'.','b');
title('Hamming windowed impulse responses N=81');
ylabel('h[n]');
xlabel('n');



[HH,w]=freqz(hH,1,4096);    %�Ӻ�������ĵ�ͨ�˲���Ƶ����Ӧ
subplot(2,1,2);
plot(w/pi,20*log10(abs(HH)));
title('Hamming Magnitude Response N=81');
ylabel('|H(��)|');
xlabel('��/��');

grid on;
set(gca,'YTickMode','manual','YTick',[-150 -100 -50 -3])

Mp=max(abs(HH))-1                                %ͨ���Ĳ����� ����ʽ10.2.3
Ap=20*log10((1+Mp)/(1-Mp))


