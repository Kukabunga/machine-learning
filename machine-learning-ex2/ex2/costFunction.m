function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

dot_product = X * theta;

J = 1 / m * sum ( dot_product, y);

for i = 1: size(theta)(1)
  grad(i) = 1 / m * grad_sum( dot_product, y, X, i );
endfor

% =============================================================

end


function s = sum(p, y)
  s = 0;
  for i = 1: length(y)
    s += -y( i ) * log( sigmoid( p( i ) ) ) - ( 1 - y( i ) ) * log ( 1 - sigmoid( p( i ) ) );
  endfor
endfunction


function s = grad_sum( p, y, X, index)
  s = 0;
  for i = 1: length(y)
    s += ( sigmoid( p ( i ) ) - y( i ) ) * X(:, index)(i);
  endfor
endfunction























