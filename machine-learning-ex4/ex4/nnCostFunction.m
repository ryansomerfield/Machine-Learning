function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y_oneHot = zeros(m, num_labels);
for i=  1 : m
  y_oneHot(i,y(i)) = 1;
endfor
y_oneHot;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

%insert the column of ones into the inputs
X = [ones(size(X, 1), 1) X];
z2 = Theta1 * X';
a2 = sigmoid(z2)';

%insert the column of ones into the a2
a2 = [ones(size(a2, 1), 1) a2];
z3 = Theta2 * a2';
a3 = sigmoid(z3)';


%y_oneHot(1,:)
regSum = 0;
jSum = 0;
for i=1:m
  for j=1:num_labels
    jSum = jSum + (y_oneHot(i,j)*log(a3(i,j))+(1-y_oneHot(i,j))*log(1-a3(i,j)));
  endfor
endfor
jSum = -1/m * jSum;



for i=1:hidden_layer_size
  for j=2:input_layer_size+1
    regSum = regSum + Theta1(i, j)^2;
  endfor
endfor
for i=1:num_labels
  for j=2:hidden_layer_size+1
    regSum = regSum + Theta2(i, j)^2;
  endfor
endfor

regSum = lambda/(2*m) * regSum;
J = jSum+regSum;

%J = -1/(m) * sum(sum((y_oneHot*log(a3')-(1-y_oneHot)*log((1-a3)'))));


%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
delta1 = 0;
delta2 = 0;
for t=1:m
  %1x401
  a1 = X(t,:);
  %25x1
  z2 = Theta1 * a1';
  %1x25
  a2 = sigmoid(z2)';

  %insert the one into the a2
  %1x26
  a2 = [1 a2];
  %10x1
  z3 = Theta2 * a2';
  %1x10
  a3 = sigmoid(z3)';

  %10x1
  d3 = (a3 - y_oneHot(t, :))'; 
  %25x1
  d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2);
  %10x26
  delta2 = delta2+d3*a2;
  %25x401
  delta1 = delta1+d2*a1;
  
  
endfor




% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
theta1mask = ones(size(Theta1));
theta2mask = ones(size(Theta2));
theta1mask(:,1) = 0;
theta2mask(:,1) = 0;

regDelta1 = lambda/m * theta1mask.*Theta1;
regDelta2 = lambda/m * theta2mask.*Theta2;



Theta1_grad = 1/m*delta1+regDelta1;
Theta2_grad = 1/m*delta2+regDelta2;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
