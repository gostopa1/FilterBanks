infile=['./sounds/darkclean.wav'];
outfile=['./sounds/darkdirt.wav']

%infile=['./piano_descending.wav'];
gap=1;
[inwav,fs]=audioread(infile); if(size(inwav,2)>1), inwav=inwav(:,1); end; inwav=inwav(1:gap:end); inwav=inwav/max(abs(inwav));
[outwav,fs]=audioread(outfile); if(size(outwav,2)>1), outwav=outwav(:,1); end; outwav=outwav(1:gap:end); outwav=outwav/max(abs(outwav));
outwav=real(ifft(circshift(fft(inwav),1000)));
fs=fs/gap;
if ~exist('Nins')
    Nins=64;
end
Nouts=1;
Nouts=Nins;


sampli=1;
gap=1;
minimumlength=min([length(inwav) length(outwav)])
x=zeros(floor((minimumlength-Nins)/gap),Nins);
y=zeros(floor((minimumlength-Nins)/gap),Nouts);

for i=(Nins+1):gap:minimumlength
     x(sampli,:)=inwav((i-Nins):(i-1));
     
     y(sampli,:)=outwav(i-1);
     %y(sampli,:)=outwav((i-Nins):(i-1));
     sampli=sampli+1;
end

t = 0:1/fs:1-1/fs;
t=t(1:Nins);
%
%f=[0 5 10 20 30 40 60:10:4000 4000:1000:(fs/2)];
%f=[ 440 880]; f=[f f];

if ~exist('nonotes')
    nonotes=24;
end
f1=110.*2.^((1:nonotes)./12);
%f=[1:20 f]; 
f=[f1 f1];
f=(1:(Nins))*(fs/Nins); f1=f(1:Nins/2); %% FFT bins 

Wsin=sin(2*pi*t'*f(1:(length(f)/2)));
Wcos=cos(2*pi*t'*f(1:(length(f)/2)));
W=Wcos;
W=[Wsin Wcos];
W(:,1:2:end)=Wsin;
W(:,2:2:end)=Wcos;
W=W.*repmat(hanning(Nins),1,length(f));

%%
o(1,:,:)=(eye(length(f)/2));
o(2,:,:)=(eye(length(f)/2));
Wsum=(reshape(o,length(f),length(f)/2));

