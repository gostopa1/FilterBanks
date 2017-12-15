function resound=pass_sound_through_model(model,sound);
%%
Nins=model.layersizes(1);

sampli=1;
gap=1;
minimumlength=length(sound);
x=zeros(floor((minimumlength-Nins)/gap),Nins);


for i=(Nins+1):gap:minimumlength
     x(sampli,:)=sound((i-Nins+1):(i));
     
     sampli=sampli+1;
end

[~,resound]=forwardpassing_nolr(model,x);

end