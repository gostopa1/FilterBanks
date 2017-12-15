function [error,dedout,errpersample]=quadratic_cost(model)
%error=sum(mean(0.5*(model.layers(end).out-model.target).^2));

errpersample=-(model.layers(end).out.*model.target);

error=sum(mean(errpersample))+model.l2/(2*model.batchsize)*sum(model.allweights(:).^2)+model.l1/(model.batchsize)*sum(abs(model.allweights(:)));

dedout=-(model.target).*sign(model.layers(end).out);

end