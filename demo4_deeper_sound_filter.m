%% Creating dataset
clear
addpath(genpath('../newer/DeepNNs\/'))
Nins=10
make_sound_data
test_data=x;
test_data=x;
x_test=x;
model.x_test=x;
y_test=y;
%% Model Initialization
clear model

layers=[2];

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
%lr=0.01; activation='linact';
%lr=0.01; activation='relu';
%lr=0.05; activation='logsi';
model.batchsize=200;
model.layersizes=[layers];
model.layersizesinitial=model.layersizes;

model.target=y;
model.epochs=10000;
model.update=500;
model.l2=0.1;
model.l1=0.1;
model.stopthres=0.00000;

model.errofun='quadratic_cost';
%model.errofun='cross_entropy_cost';

for layeri=1:(length(layers)-1)
    
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)];
    %model.layers(layeri).W=(randn(layers(layeri),layers(layeri+1))-0.5)/10;
    
    model.layers(layeri).W=1*(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
    %model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/10;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
end
model.layers(1).blr=0;
%model.layers(1).W(:)=1/noins;
%model.layers(1).W=repmat((-1).^(1:noins)',1,model.layersizes(2));
%model.layers(1).W=repmat((-1).^(1:noins)',1,model.layersizes(2));
%model.layers(1).W(3:end,:)=0;
    
layeri=1;
model.layers(layeri).W=1*(zeros(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));

crossfreq=5000;
bpFilt = designfilt('lowpassfir','FilterOrder',Nins-1, 'CutoffFrequency',crossfreq , 'SampleRate',fs);
bp1 = bpFilt.Coefficients;
model.layers(1).W(:,1)=bp1;

bpFilt = designfilt('highpassfir','FilterOrder',Nins-1, 'CutoffFrequency',crossfreq , 'SampleRate',fs);
bp2= bpFilt.Coefficients;
model.layers(1).W(:,2)=bp2;

%model.layers(2).W(:,:)=0;
%model.layers(2).W(2,:)=1;

%% Model training

clear error
for layeri=1:(length(model.layers))
    model.layers(layeri).nonzeroinds=find(model.layers(layeri).W~=0);
end
model.epoch=1;
epoch=1;
    model=vectorize_all_weights(model);

[model,out2(:,:,model.epoch)]=forwardpassing(model,model.x);
[model.error(epoch),dedout]=feval(model.errofun,model);

show_network(model)

%% Visual evaluation

model.test=0;
[model,out_test]=forwardpassing(model,[test_data]);
factor=15;

dur=5;
sampledur=fs*dur;

soundsc(out_test(1:sampledur),fs)
% return
% soundsc(inwav(1:sampledur),fs)
% pause(dur)
% soundsc(outwav(1:sampledur),fs)
% pause(dur)
% soundsc(out_test(1:sampledur),fs)

figure(1);clf
subplot(6,1,[1 3])
hold on
plot(bp1,'LineWidth',2)
plot(bp2,'LineWidth',2)
legend({'High','Low'},'Location','SouthOutside')
subplot(6,1,[4:6])
show_network_local
%set(gcf,'PaperPosition',[0 0 400 600]/40); print(['./figures/' num2str(Nins) 'hilowfilter' sprintf('w%2.2f',model.layers(1).W(1)) '.png'],'-dpng','-r500')
audiowrite(['./result_sounds/' num2str(Nins) 'hilowfilter.wav'],out_test,fs)

%%


