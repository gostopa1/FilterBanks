clear
close all
lw=30;

ind=1;

entry(ind).description='Sibelius Academy - Bachelor in Music Technology';
entry(ind).stend=[2013.8 2017.5];
ind=ind+1;

entry(ind).description='Computer Engineering'
entry(ind).stend=[2005.8 2011.5];
ind=ind+1;

entry(ind).description='Diploma Thesis Biomedicum Helsinki'
entry(ind).stend=[2010.8 2011.5];
ind=ind+1;

entry(ind).description='Aalto University - PhD student'
entry(ind).stend=[2012 2018.8];
ind=ind+1;

entry(ind).description='Military Service - Finland'
entry(ind).stend=[2016.5 2017];
ind=ind+1;

entry(ind).description='DJ in bars and clubs'
entry(ind).stend=[2005 2010];
ind=ind+1;

entry(ind).description='Live gigs'
entry(ind).stend=[2005 2010];
ind=ind+1;





for entri=1:length(entry)
    if length(entry(entri).stend)>1
       line([entry(entri).stend(1) entry(entri).stend(2)],[entri entri],'Color',rand(3,1),'LineWidth',lw) 
       text(mean(entry(entri).stend),entri+0.,entry(entri).description,'HorizontalAlignment','center','VerticalAlignment','middle') 
    end
end

hAxes = gca;     %Axis handle
%Changing 'LineStyle' to 'none'
%hAxes.XRuler.Axle.LineStyle = 'none';  
set(gca,'YTick','')
set(gca,'XTick',2005:2020)
hAxes.YRuler.Axle.LineStyle = 'none';

axis([2005 2021 -0 length(entry)+1])