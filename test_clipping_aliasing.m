fs=44100;
t = 0:1/fs:1-1/fs;
L=round(fs/5);
t=t(1:L);

f1=5000;
x=sin(2*pi*f1*t);

limit=0.9
%x(x>limit)=limit;
%x(abs(x)>limit)=limit;
inds=abs(x)>limit
x(inds)=sign(x(inds))*limit;

sound(0.5*x,fs)

plot(t,x)

%%
%f=(1:(Nins))*(fs/Nins); f1=f(1:Nins/2); %% FFT bins 
f = fs*(0:(L/2))/L;
%f = fs*(0:(L-1))/L;
Y=fft(x);
%Y=sum(cos(2*pi*t'*f)-i*sin(2*pi*t'*f),1)';
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
%P1=abs(Y/L);
P1(2:end-1) = 2*P1(2:end-1);
%%





plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')


%%

