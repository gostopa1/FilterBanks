[y,fs]=audioread('gtr.wav');
secs=5;
y1=y(1:(fs*secs),1);
y1=y1/max(abs(y1));
%%
filtorder=4410; display(['This causes delay of ' sprintf('%3.2f',filtorder*1000/fs) ' milliseconds'])
b = fir1(100,[0.35 0.65]);
%b = fir1(34,0.48,'high',chebwin(35,30));
freqz(b,1,[],fs)


y2=filter(b,1,y1); y2=y2/max(abs(y2));
y3=y1-y2;
y3=waveshape(y2); y3=y3-mean(y3); y3=y3/max(abs(y3));
%y2=y3+y2;
y2=y3;

soundsc(y2,fs)



%%


x=-1:0.01:1;


plot(x,waveshape(x))


%y=0.5*x+0.3*x.^2;

plot(x,y)