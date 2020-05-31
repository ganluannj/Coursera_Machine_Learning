function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
Xtheta=X*theta;
Sig=sigmoid(Xtheta);
J=-1/m*(y'*log(Sig)+(1-y)'*log(1-Sig)) + ...
        lambda/(2*m)*(theta'*theta-theta(1)^2);

% for calculating gradient we let theta(1) be 0
thetamodi=theta;
thetamodi(1)=0;        
grad=1/m*X'*(Sig-y)+lambda/m*thetamodi;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
