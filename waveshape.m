function y=waveshape(x)

%y=1.5*x - 0.5*x.^3; % Very soft clipping
%y=x./(1+abs(x)); %% Good

%y=sin(2*x).^3;
y=16*x.^5-20*x.^3+5*x;
end

%%
