function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

% From the notes: To multiply two matrices, the number of columns of the first matrix must equal the number of rows of the second matrix. (X = 97x1, theta = 1x2)

sumSquaredErrors = sum((X * theta - y) .^ 2);

J = 1 / (2 * m) * sumSquaredErrors;

% =========================================================================

end
