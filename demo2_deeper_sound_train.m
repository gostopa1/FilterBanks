%% Creating dataset
addpath(genpath('../DeepNNs/'))
Nins=10;
make_sound_data
test_data=x;
x_test=x;

model.x_test=x;
y_test=y;
%% Model Initialization
clear model

layers=[];

noins=size(x,2);
noouts=size(y,2);
model.x=x;
model.y=y;
model.fe_update=100000;
model.fe_thres=0.000;
model.N=size(x,1);
layers=[noins layers noouts];
lr=0.01; activation='softsign';
%lr=0.01; activation='sincact';
%lr=0.01; activation='linact';
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
%model.layers(1).blr=0;
%model.layers(2).activation='softsign';
%model.layers(2).activation='linact';
%model.layers(layeri).lr=lr; model.layers(layeri).activation='softmaxact';


%% Model training

clear error
for layeri=1:(length(model.layers))
    model.layers(layeri).nonzeroinds=find(model.layers(layeri).W~=0);
end
model.epoch=1;

[model,out2(:,:,model.epoch)]=forwardpassing(model,model.x);

model.fnzl=find_first_non_zero_layer(model);
if (length(model.layers)>1) && (model.fnzl==length(model.layers))
    display('All learning rates are zero. The model will be identical! No point continuing!');
    return;
end


for epoch=1:model.epochs
    stind=randi(model.N-model.batchsize);
    
    batchinds=stind:(stind+model.batchsize-1);
    %batchinds=randperm(model.N,model.batchsize);    
    model.target=model.y(batchinds,:);
    model.epoch=epoch;
    if mod(epoch,model.update)==0
        % Now it is time to show an update of the network.
        
        plot(model.error)
        drawnow
        % If there is a test set, calculate the accuracy for it
        if sum(isfield(model,{'x_test','y_test'}))==2
            
            epochtemp=model.epoch; model.epoch=0;
            [~,out_test]=forwardpassing(model,model.x_test(:,model.layers(1).inds)); get_perf(out_test,model.y_test);
            model.epoch=epochtemp;
            y_test_str=[' - Test: ' sprintf('%2.2f', get_perf(out_test,model.y_test))];
        else
            y_test_str='';
        end
        [~,out_train]=forwardpassing(model,model.x);
        display(['Epoch: ' num2str(epoch) ' - Training: ' sprintf('%2.2f',get_perf(out_train,model.y)) y_test_str]);
    end
    % Should I perform feature elimination
    if mod(epoch,model.fe_update)==0
        model=feature_elimination(model);
    end
    
    %%Forward passing
    model=vectorize_all_weights(model);
    
    %[model,out(:,:,epoch)]=forwardpassing_nolr(model,model.x(batchinds,:));
    [model,out(:,:,epoch)]=forwardpassing_nolr(model,model.x(batchinds,:));
    
    [model.error(epoch),dedout]=feval(model.errofun,model);
    
    
    for layeri=(length(model.layers)):-1:model.fnzl
        clear dedw dedb
        ins=model.layers(layeri).Ws(1);
        outs=model.layers(layeri).Ws(2);
        
        if layeri==length(model.layers)
            % For the last layer the gradient is calculated as
            % dE/dw = dE/dout * dout/dnet * dnet/dw
            dnetdw=repmat(model.layers(layeri).X,1,1,outs); % dnet/dw
            
            %dedb=(dedout(batchinds,:).*model.layers(layeri).doutdnet(batchinds,:))'; % dE/db = dE/dout * dout/dnet
            dedb=(dedout.*model.layers(layeri).doutdnet)'; % dE/db = dE/dout * dout/dnet
            dedw=permute(repmat(dedb,1,1,ins),[2 3 1]).*dnetdw; % dE/dw = dE/dout * dout/dnet * dnet/d
        else
            
            dedb=permute(mean(permute(repmat(model.layers(layeri+1).grad,1,1,outs),[2 3 1]).*permute(repmat(model.layers(layeri+1).W,1,1,model.batchsize),[3 1 2]),3).*model.layers(layeri).doutdnet,[2 1]);
            dedw=permute(repmat(dedb,1,1,ins),[2 3 1]).*repmat(model.layers(layeri).X,1,1,outs);
            
        end
        %layeri
        %size(dedb)
        model.layers(layeri).grad=dedb;
        
        
        l1part=-model.layers(layeri).lr*(model.l1/model.batchsize).*sign(model.layers(layeri).W);
        l2part=-model.layers(layeri).lr*(model.l2/model.batchsize).*model.layers(layeri).W;
        regularization_term=model.layers(layeri).W+l1part+l2part;
        mdedw=permute(mean(dedw,1),[2 3 1]);
        
        %model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr(model.layers(layeri).nonzeroinds).*mdedw(model.layers(layeri).nonzeroinds);
        model.layers(layeri).W(model.layers(layeri).nonzeroinds)=regularization_term(model.layers(layeri).nonzeroinds)-model.layers(layeri).lr.*mdedw(model.layers(layeri).nonzeroinds);
        model.layers(layeri).B=model.layers(layeri).B-model.layers(layeri).blr.*mean(dedb,2);
        
        
    end
    
    
end
show_network

figure(80)
clf
subplot(3,1,1)
plot(x(batchinds))
hold on
%plot(erper,'r')
subplot(3,1,2)
%plot(erper)


%save_figure
%% Visual evaluation
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

figure(5)
clf
hold on
plot(outwav,'k')
plot(out_test,'b')
legend({'Original','Modeled'})


dur=5;
sampledur=fs*dur;

soundsc(out_test(1:sampledur),fs)
return
soundsc(inwav(1:sampledur),fs)
pause(dur)
soundsc(outwav(1:sampledur),fs)


%%

