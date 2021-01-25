function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

% Cost
left = -y' * log(h);
right = (1 - y)' * log(1 - h);
J = (1 / m) * (left - right);
theta(1, :) = 0;
J_reg = (lambda / (2 * m)) * (theta' * theta);
J = J + J_reg;

% Gradient
errors = h - y;
grad_reg = (lambda / m) * theta;
grad = (1 / m) * (X' * errors) + grad_reg;


% =============================================================

end
