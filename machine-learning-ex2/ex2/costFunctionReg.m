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

dot_product = X * theta;

J = 1 / m * sum_cost ( dot_product, y, lambda, theta);
J += lambda / ( 2 * length(y) ) * sum(theta(2:end).^ 2);

for i = 1: size(theta)(1)
  t = 0;
  if ( i != 1)
    t = lambda / m * theta(i);
  endif
  grad(i) = 1 / m * grad_sum( dot_product, y, X, i ) + t;
endfor


% =============================================================

end


function s = sum_cost(p, y, lambda, theta)
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

