%����Ƶ��ѡ���������˲���������ָ�꣬����������λFIR�����˲��������뵥λ�弤��Ӧ

clear all;
close all;
clc;

N1=521;N2=121;                            % ���ĳ���
M1=(N1-1)/2; M2=(N2-1)/2;                % Ⱥʱ��
Wp=0.3*pi;                               % ͨ����ֹƵ��
Ws=0.4*pi;                               % �����ֹƵ��
Wc=(Wp+Ws)/2;


%��λ�弤��Ӧ�Ӿ��δ�
% N1=51ʱ
figure(1);

n=0:N1-1;
hR=Wc/pi*sin(Wc*(n-M1+eps))./(Wc*(n-M1+eps)); %�Ӿ��δ���λ�弤��Ӧ
subplot(2,1,1);
stem(n,hR,'.','b');
title('Rectangular windowed impulse responses N=51');
ylabel('h[n]');
xlabel('n');


subplot(2,1,2);
[HR,w]=freqz(hR,1,8192);    %�Ӿ��δ���ĵ�ͨ�˲���Ƶ����Ӧ
plot(w/pi,20*log10(abs(HR)));
axis([0,1,-150 20]);
title('Rectangular Magnitude Response N=51');
ylabel('|H(��)|');
xlabel('��/��');
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])
Mp=max(abs(HR))-1                                %ͨ���Ĳ����� ����ʽ10.2.3
Ap=20*log10((1+Mp)/(1-Mp))


figure(2);

n=0:N2-1;
hR=Wc/pi*sin(Wc*(n-M2+eps))./(Wc*(n-M2+eps)); %�Ӿ��δ���λ�弤��Ӧ
subplot(2,1,1);
stem(n,hR,'.','b');
title('Rectangular windowed impulse responses N=121');
ylabel('h[n]');
xlabel('n');


subplot(2,1,2);
[HR,w]=freqz(hR,1,8192);    %�Ӿ��δ���ĵ�ͨ�˲���Ƶ����Ӧ
plot(w/pi,20*log10(abs(HR)));
axis([0,1,-150 20]);
title('Rectangular Magnitude Response N=121');
ylabel('|H(��)|');
xlabel('��/��');
grid on;
set(gca,'YTickMode','manual','YTick',[-100 -50 -3])

Mp=max(abs(HR))-1                                %ͨ���Ĳ����� ����ʽ10.2.3
Ap=20*log10((1+Mp)/(1-Mp))
