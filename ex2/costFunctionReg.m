function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = size(X)(2); % number of independent variables

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h_X = sigmoid(X * theta);
J_1 = y' * log(h_X);
J_2 = (1 - y)' * log(1 - h_X);
J_3 = lambda .* sum(theta(2:n).^2);
J = (-J_1 - J_2)./m + J_3./(2*m);

grad = X' * (sigmoid(X * theta) - y) ./ m;
gradReg = (lambda/m) .* theta;
gradReg(1) = 0
grad = grad + gradReg;

% =============================================================

end
