

infile=['./sounds/darkclean.wav'];


outfile=['./sounds/darkdirt.wav']
if exist('gap')~=1
    gap=2;
end

[inwav,fs]=audioread(infile); if(size(inwav,2)>1), inwav=inwav(:,1); end; inwav=inwav(1:gap:end); inwav=inwav/max(abs(inwav));
[outwav,fs]=audioread(outfile); if(size(outwav,2)>1), outwav=outwav(:,1); end; outwav=outwav(1:gap:end); outwav=outwav/max(abs(outwav));
fs=fs/gap;
if ~exist('Nins')
    Nins=10;
end
Nouts=1;
Nouts=Nins;

sampli=1;
samplegap=1;
minimumlength=min([length(inwav) length(outwav)])
x=zeros(floor((minimumlength-Nins)/samplegap),Nins);
y=zeros(floor((minimumlength-Nins)/samplegap),Nouts);

for i=(Nins+1):samplegap:minimumlength
     x(sampli,:)=inwav((i-Nins+1):(i));
     %y(sampli,:)=outwav(i);
     y(sampli,:)=outwav((i-Nins+1):(i));
     sampli=sampli+1;
end