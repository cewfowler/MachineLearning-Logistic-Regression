function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% Initial cost function + gradient
[J, grad] = costFunction(theta, X, y);

% Add regularization parameters (don't use theta0)
J = J + lambda/(2*m) * sum(theta(2:size(theta)) .^ 2);
grad(2:size(grad)) = grad(2:size(grad)) + lambda/m * theta(2:size(grad));






% =============================================================

end
