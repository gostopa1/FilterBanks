clear

infile=['./sounds/darkclean.wav'];


outfile=['./sounds/darkdirt.wav']

gap=2;
[inwav,fs]=audioread(infile); if(size(inwav,2)>1), inwav=inwav(:,1); end; inwav=inwav(1:gap:end); inwav=inwav/max(abs(inwav));
[outwav,fs]=audioread(outfile); if(size(outwav,2)>1), outwav=outwav(:,1); end; outwav=outwav(1:gap:end); outwav=outwav/max(abs(outwav));
fs=fs/gap;
if ~exist('Nins')
    Nins=10;
end
Nouts=2*Nins;

sampli=1;
gap=1;
minimumlength=min([length(inwav) length(outwav)])
x=zeros(floor((minimumlength-Nins)/gap),Nins);
y=zeros(floor((minimumlength-Nins)/gap),Nouts);

for i=(Nins+1):gap:minimumlength
     x(sampli,:)=inwav((i-Nins):(i-1));
     tempf=fft(x(sampli,:));
     y(sampli,:)=[real(tempf)  imag(tempf)];
     sampli=sampli+1;
end