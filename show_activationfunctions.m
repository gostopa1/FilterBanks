clear
n=1000;
xlim=5;
x=linspace(-xlim,xlim,n);
w=1;
b=0;
[res1,der1]=softsign(x,w,b)
ind=1;
[res(ind,:),der(ind,:)]=softsign(x,w,b); ind=ind+1;
[res(ind,:),der(ind,:)]=logsi(x,w,b); ind=ind+1;
[res(ind,:),der(ind,:)]=relu(x,w,b); ind=ind+1;
[res(ind,:),der(ind,:)]=tanhact(x,w,b); ind=ind+1;
%[res(ind,:),der(ind,:)]=linact(x,w,b); ind=ind+1;
[res(ind,:),der(ind,:)]=sincact(x,w,b); ind=ind+1;
%[res(ind,:),der(ind,:)]=sinact(x,w,b); ind=ind+1;


figure(1)
subplot(1,2,1)
plot(x,res','LineWidth',3)



axis([-xlim xlim -xlim xlim])
h1=legend({'Softsign','Logistic','ReLU','Tanh','Sinc'},'Location','EastOutside')
set(h1,'Visible','Off')
xlabel('Input')
ylabel('Output')
title('Activation functions')
axis([-xlim xlim -xlim xlim])
subplot(1,2,2)
plot(x,der','LineWidth',3)
title('Activation function derivatives')
h1=legend({'Softsign','Logistic','ReLU','Tanh','Sinc'},'Location','EastOutside')
axis([-xlim xlim -xlim xlim])

set(gcf,'PaperPosition',[0 0 1300 400]/40); set(gcf,'Position',[0 0 1300 400]); print(['./figures/actfun.png'],'-dpng','-r300')

