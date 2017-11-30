%% Creating dataset
addpath(genpath('../newer/DeepNNs/'))
Nins=100 ;
make_sound_data
test_data=x;
x_test=x;

model.x_test=x;
y_test=y;
%% Model Initialization
clear model

layers=[ ];

noins=size(x,2);
noouts=size(y,2);
model.x=x;
model.y=y;
model.fe_update=100000;
model.fe_thres=0.000;
model.N=size(x,1);
layers=[noins layers noouts];
lr=0.01; activation='softsign';
%lr=0.01; activation='linact';

%lr=0.01; activation='sincact';

%lr=0.01; activation='relu';
%lr=0.05; activation='logsi';
%model.batchsize=2000;
model.batchsize=1000;
model.layersizes=[layers];
model.layersizesinitial=model.layersizes;

model.target=y;
model.epochs=10000;
model.update=500;
model.l2=0.01;
model.l1=0.0;
model.stopthres=0.00000;

%model.errofun='quadratic_cost';
model.errofun='cross_entropy_cost';

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

%model.layers(1).W(1)=8
figure(1);clf
show_network_local
set(gcf,'PaperPosition',[0 0 400 400]/40); print(['./figures/' num2str(Nins) 'vs1net' sprintf('w%2.2f_',model.layers(1).W(1)) activation '.png'],'-dpng','-r300')
[~,out_test]=forwardpassing(model,x);
out_test=zscore(out_test);

dur=5;
sampledur=fs*dur;

%soundsc(inwav(1:sampledur),fs)
%pause
soundsc(out_test(1:sampledur),fs)
audiowrite(['./result_sounds/' num2str(Nins) 'vs1net' sprintf('w%2.2f',model.layers(1).W(1)) '.wav'],out_test,fs)
%%
figure(5)
clf
subplot(2,1,1)
plot(inwav,'b')
title('Input signal')
subplot(2,1,2)
plot(out_test)
title('Output signal')