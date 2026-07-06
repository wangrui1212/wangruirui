clear
[filename, pathname]= uigetfile({'*.dat'},'Seleck a File');
if ((isequal(filename,0))||(isequal(pathname,0)))
    return
end
fpath=fullfile(pathname,filename);
f=fopen(fpath,'r');
A=fread(f,'float');

%%------------------把A数组读成矩阵
A = A.';
Header = A(1:64);   %包头
Data = A(65:end); %数据
fs = A(11);   %采样率
time = A(18); %采集时间(单位：秒)
pulses = fs * time; %脉冲个数
data_length = floor( length(Data)/pulses); %数据长度

B = zeros(pulses, data_length);
for i = 1:pulses
    B(i,:) = Data((1:data_length) + data_length*(i-1));
end

loc_start = 1;
loc_end = data_length;
ttt = size(B);
t=(1:ttt(1))/fs;

figure
hold on 
for i=loc_start:loc_end
    plot(t,B(:,i)+i);
end