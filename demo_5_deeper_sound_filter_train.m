%% Creating dataset
clear
addpath(genpath('../newer/DeepNNs/'))
Nins=15; gap=1;
make_sound_data
test_data=x;
test_data=x;
x_test=x;
model.x_test=x;
y_test=y;
%% Model Initialization
clear model

layers=[18];

noins=size(x,2);
noouts=size(y,2);
model.x=x;
model.y=y;
model.fe_update=100000;
model.fe_thres=0.000;
model.N=size(x,1);
layers=[noins layers noouts];
lr=0.01; activation='softsign';
%lr=0.01; activation='tanhact';
%lr=0.005; activation='linact';
%lr=0.01; activation='relu';
%lr=0.05; activation='logsi';
model.batchsize=1000;
model.layersizes=[layers];
model.layersizesinitial=model.layersizes;

model.target=y;
model.epochs=1000;
model.update=100;
model.l2=0.00;
model.l1=0.0;
model.stopthres=0.00000;

model.errofun='quadratic_cost';
model.errofun='cross_entropy_cost';

for layeri=1:(length(layers)-1)
    
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=0;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)];
    %model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/10;
    
    model.layers(layeri).W=1*(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
    %model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/10;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
end
model.layers(1).lr=0;
%model.layers(1).W(:)=1/noins;
%model.layers(1).W=repmat((-1).^(1:noins)',1,model.layersizes(2));
%model.layers(1).W=repmat((-1).^(1:noins)',1,model.layersizes(2));
%model.layers(1).W(3:end,:)=0;
    
% layeri=1;
%model.layers(layeri).W=1*(zeros(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));

%%
%model.layers(1).W(:,1)=b;
%bp1= fir1(n,[0.5 0.6],'bandpass');
basefreq=22.5;
octavesteps=2;
for filteri=1:model.layersizes(2)
%model.layers(1).W(:,2)=bp1;
%bp2 = fir1(n,[freqstep*(filteri) freqstep*(filteri+1)]*(pi/fs),'bandpass');

%bpFilt = designfilt('bandpassfir','FilterOrder',Nins-1, 'CutoffFrequency1',freqstep*(filteri) ,'CutoffFrequency2',freqstep*(filteri+1) , 'SampleRate',fs);
bpFilt = designfilt('bandpassfir','FilterOrder',Nins-1, 'CutoffFrequency1',basefreq*2^((filteri-1)/octavesteps) ,'CutoffFrequency2',basefreq*2^((filteri)/octavesteps) , 'SampleRate',fs);
bp2 = bpFilt.Coefficients;
basefreq*2^((filteri-1)/octavesteps)
model.layers(1).W(:,filteri)=bp2;
imps(filteri,:)=bp2;
end

figure(5)
clf
%fvtool(bpFilt)
show_network_local

set(gcf,'PaperPosition',[0 0 500 500]/40); print(['./figures/' num2str(Nins) 'filterbank' sprintf('w%2.2f',model.layers(1).W(1)) '.png'],'-dpng','-r500')


%set(gcf,'PaperPosition',[0 0 400 400]/40); print(['./figures/' num2str(Nins) 'filterbank' sprintf('w%2.2f',model.layers(1).W(1)) '.png'],'-dpng','-r500')

%plot(imps)
%imagesc(imps)

%% Model training

clear error
for layeri=1:(length(model.layers))
    model.layers(layeri).nonzeroinds=find(model.layers(layeri).W~=0);
end
model.epoch=1;
epoch=1;
model=vectorize_all_weights(model);

%%

dur=3;
sampledur=fs*dur;
[~,out_test]=forwardpassing(model,[test_data]);

%model.layers(2).W=1*(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
%model.layers(2).W=0*(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
%model.layers(2).W(5)=1;
soundsc(out_test(1:sampledur),fs)
%return
%%


model=model_train_fast(model);
show_network(model)

% model2=model;
model.test=0;
[model,out_test]=forwardpassing(model,[test_data]);
factor=15;

%axis off
%box off
subplot(4,1,4)
plot(model.error)
xlabel('Epoch')

ylabel('Error')


soundsc(out_test(1:sampledur,1),fs)
% return
% soundsc(inwav(1:sampledur),fs)
% pause(dur)
% soundsc(outwav(1:sampledur),fs)
% pause(dur)
% soundsc(out_test(1:sampledur),fs)

%%

figure(5)
clf
hold on
plot(outwav)
plot(out_test)

