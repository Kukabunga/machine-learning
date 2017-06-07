function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for i = 1:rows(z)
  if (columns(z) > 0)
    for j = 1:columns(z)
      g(i, j) = calc(z(i,j));
    endfor
  else
     g(i) = calc(z(i));
  endif
endfor
% =============================================================

end



function result = calc(n)
  result = 1 / ( 1 + exp(-n) );
endfunction