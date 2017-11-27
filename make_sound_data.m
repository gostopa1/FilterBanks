clear

infile=['./sounds/gtrriff_clean.wav'];


outfile=['./sounds/gtrriff_dist.wav']

gap=10;
[inwav,fs]=audioread(infile); if(size(inwav,2)>1), inwav=inwav(1:gap:end,1); end; inwav=inwav/max(abs(inwav));
[outwav,fs]=audioread(outfile); if(size(outwav,2)>1), outwav=outwav(1:gap:end,1); end;outwav=outwav/max(abs(outwav));
fs=fs/gap;
Nins=2001;
Nouts=1;

sampli=1;
gap=1;
minimumlength=min([length(inwav) length(outwav)])
x=zeros(floor((minimumlength-Nins)/gap),Nins);
y=zeros(floor((minimumlength-Nins)/gap),Nouts);

for i=(Nins+1):gap:minimumlength
     x(sampli,:)=inwav((i-Nins):(i-1));
     y(sampli,:)=outwav(i);
     sampli=sampli+1;
end