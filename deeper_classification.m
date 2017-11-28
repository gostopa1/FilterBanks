%% Creating dataset

clear
addpath(genpath('../DeepNNs/'))

x = [1 1; 1 0; 0 1; 0 0];
y(:,2) = [1; 0; 0; 1];
y(:,1) = [0; 1; 1; 0];
N=size(x,1);

N=1000; x=rand(N,2); y=xor(x(:,1)<0.5,x(:,2)<0.5); y(:,2) = 1 - y(:,1);

x=zscore(x,[],1);

%y=zscore(y,[],1)

[x_test,y_test]=meshgrid(0:0.1:1,0:0.1:1);

%test_data=[x_test(: ) y_test(:)];
test_data=zscore([x_test(: ) y_test(:)],1);

x=x/max(x(:)); test_data=test_data/max(abs(test_data(:)));

%% Model Initialization
clear model

model.x=x;
model.y=y;
model.N=N;
model.fe_update=100000;

layers=[];
model.batchsize=200;
noins=size(x,2);
noouts=size(y,2);

layers=[noins layers noouts];
lr=0.1; activation='tanhact';
lr=0.1; activation='softsign';
%lr=0.1; activation='logsi';
model.l2=0;
model.l1=0;
model.layersizes=[layers];
model.layersizesinitial=model.layersizes;
model.target=y;
model.epochs=1000;
model.update=100;
model.errofun='quadratic_cost';
model.errofun='cross_entropy_cost';
for layeri=1:(length(layers)-1)
    model.layers(layeri).lr=lr;
    model.layers(layeri).blr=lr;
    model.layers(layeri).Ws=[layers(layeri) layers(layeri+1)]
    
    model.layers(layeri).W=1*(randn(layers(layeri),layers(layeri+1)))*sqrt(2/(model.layersizes(layeri)+model.layersizes(layeri+1)));
    %model.layers(layeri).B=(randn(layers(layeri+1),1)-0.5)/10;
    model.layers(layeri).B=(zeros(layers(layeri+1),1))/1;
    model.layers(layeri).activation=activation;
    model.layers(layeri).inds=1:model.layersizes(layeri); % To keep track of which nodes are removed etc
end
model.layers(layeri).lr=lr/1; model.layers(layeri).activation='softmaxact';

%% Model training

clear error

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
    
    batchinds=randperm(model.N,model.batchsize);    
    model.target=model.y(batchinds,:);
    model.epoch=epoch;
    if mod(epoch,model.update)==0
        % Now it is time to show an update of the network.
        show_network
        subplot(4,1,4)
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
%save_figure
%% Visual evaluation

model.test=0;
[model,out_test]=forwardpassing(model,[test_data]);
factor=15;
figure(1)
%clf
subplot(4,1,3)
hold on

for pointi=1:size(test_data,1)
    mat(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9)=out_test(pointi,1);
end
contour(mat,[0.25 0.5 0.75],'LineWidth',3)
test_labs=xor(x_test<0.5,y_test<0.5);
for pointi=1:size(test_data,1)
    if out_test(pointi,1)>0
        dotcol='k';
    else
        dotcol='r';
    end
    %m=plot(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9,[dotcol '.']);
    text(((x_test(pointi)+1)*10)-9,((y_test(pointi)+1)*10)-9,num2str(test_labs(pointi)));
    %set(m,'MarkerSize',abs(out_test(pointi,1))*factor+5)
end
set(gca,'XTick',[1 11],'XTickLabel',[0 1]);
set(gca,'YTick',[1 11],'YTickLabel',[0 1]);

%axis off
%box off
subplot(4,1,4)
plot(model.error)
xlabel('Epoch')

ylabel('Error')

%% Numerical evaluation (i.e. classification accuracy, etc.)

[~,indpre]=max(out(:,:,end)');

%indpre-1
[~,indtarget]=max(model.target');
perf=(sum(indpre==indtarget)/length(indpre))*100;
display([sprintf('Performance : %3.2f%%',perf)])
