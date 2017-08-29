function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
J0 = 0;
A = sigmoid(X*theta); % A: m*1
for i=1:m
    J0 = J0 + -1*y(i)*log(A(i)) - (1-y(i))*log(1-A(i));
end

J0 = J0/m;

J1 = 0;
for j=2:size(theta,1)
    J1 = J1 + theta(j)^2;
end
J1 = J1*lambda/(m*2);

J = J0 + J1;

for i=1:size(theta,1)
    grad(i) =( X(:,i)' * (A-y) )/ m;
    if i > 1
        grad(i) = grad(i) + lambda/m * theta(i);
    end
end
% =============================================================

end
