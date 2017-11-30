%% Creating dataset
clear
addpath(genpath('../newer/DeepNNs/'))
Nins=1024; nonotes=96;
make_sound_fft_data
test_data=x;
x_test=x;

model.x_test=x;
y_test=y;
%% Model Initialization
clear model
nobins=size(W,2)
%layers=[Nins 2 1 Nins Nins];
layers=[ nobins  nobins  ];
%layers=[];

noins=size(x,2);
noouts=size(y,2);
model.x=x;
model.y=y;
model.fe_update=100000;
model.fe_thres=0.000;
model.N=size(x,1);
layers=[noins layers noouts];
%lr=0.01; activation='softsign';
%lr=0.01; activation='sincact';
lr=0.01; activation='linact';
%lr=0.01; activation='relu';
%lr=0.05; activation='logsi';
%model.batchsize=2000;
model.batchsize=1000;
model.layersizes=[layers];
model.layersizesinitial=model.layersizes;

model.target=y;
model.epochs=1000;
model.update=100;
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
%model.layers(1).blr=0;f

%model.layers(2).activation='linact';
%model.layers(layeri).lr=lr; model.layers(layeri).activation='softmaxact';
%model.layers(1).activation='quadraticact';
model.layers(2).activation='linact';
%model.layers(2).activation='squarerootact';
%model.layers(2).activation='softsign';

%model=model_train_fast(model);
model.layers(1).W=W;
%model.layers(2).W=Wsum;

%model.layers(end-1).W=W(:,1:2:end)'; % If you take the last one, it will suffer from the hanning window
%model.layers(end-1).W=W'; % If you take the last one, it will suffer from the hanning window
model.layers(end).W=W(Nins/2,:)'; % If you take the last one, it will suffer from the hanning window
model.layers(end).W=W'; % If you take the last one, it will suffer from the hanning window
model.layersizes
%%
layeri=2;
for layeri=2:(length(model.layers)-2)
    %for layeri=3
    model.layers(layeri).W=1*(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
end
layeri=2;
%model.layers(4).W(1,1)=1;
model.layers(layeri).W(:)=0;
%model.layers(2).W(2,19)=2*(1/Nins);

freqshift=4;
for i=1:2:(nobins-freqshift)
    %for i=1:2:(model.layersizes(layeri+1))
    %model.layers(layeri).W(i,i)=1;
    %model.layers(layeri).W(i,i)=1;
    
    model.layers(layeri).W(i+1,mod(i+freqshift,nobins)+1)=1;
    model.layers(layeri).W(i,mod(i+freqshift-1,nobins)+1)=1;
    %nextind=mod(i+freqshift,nobins/2)+1;
    %model.layers(layeri).W(i,nextind)=1;
    
    %model.layers(layeri).W(i,i)=1;
    
    %model.layers(layeri).W(i,i)=1;
end

%figure(5); clf;
%show_layer(model, [2 3])

freqi=20:40;
freqj=round(freqi*2);

%model.layers(layeri).W(freqi,freqj)=10;

model.test=0;
[model,out_test]=forwardpassing(model,[test_data]);
sampleind=1:1000:size(x,1);


dur=5;
sampledur=fs*dur;

soundsc(sum(out_test(1:sampledur,:),2),fs)
% return
% soundsc(inwav(1:sampledur),fs)
% pause(dur)
% soundsc(outwav(1:sampledur),fs)

%% Check out FFT

nf=Nins; %number of point in DTFT
Y = fft(inwav,nf);

f = fs/2*linspace(0,1,nf/2+1);
subplot(2,1,2)
plot(f,abs(Y(1:nf/2+1)));


audiowrite(['./result_sounds/' num2str(Nins) 'FFT.wav'],out_test,fs)
if Nins<20
    figure(1);clf
    show_network_local
    set(gcf,'PaperPosition',[0 0 600 300]/40); print(['./figures/' num2str(Nins) 'FFT' sprintf('w%2.2f',model.layers(1).W(1)) '.png'],'-dpng','-r300')
end
