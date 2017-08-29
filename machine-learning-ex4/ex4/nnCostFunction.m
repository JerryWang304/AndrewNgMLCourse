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
% for our 2 layer neural network\
% nn_params = [Theta1(:) ; Theta2(:)];
% Theta1 : 25*401 = hidden_layer_size * (input_layer_size + 1)
% Theta2 : 10*26 = num_labels * (hidden_layer_size + 1)
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
% y  :  m*1
% yi :  10*1
% add ones to X
X = [ones(m,1) X];
new_y = zeros(m,num_labels);
for i=1:m
    number = y(i);
    new_y(i, number) = 1;
end
% new_y : 5000(m) * 10
J = 0;
for i=1:m
    a1 = X(i,:)'; % the i-th input
    
    z2 = Theta1*a1;
    a2 = sigmoid(z2);
    a2 = [1;a2]; % add ones to hidden layer
    z3 = Theta2*a2;
    a3 = sigmoid(z3);% h_theta(x)
    for k=1:num_labels
        t = a3(k);
        J = J + -1*new_y(i,k)*log(t) - (1-new_y(i,k))*log(1-t);
    end
end
% You need to return the following variables correctly 
J = J/m;
% add regularation 
temp_theta1 = Theta1;
temp_theta1(:,1) = zeros(hidden_layer_size,1);
sum1 = sum(sum(temp_theta1 .^ 2));
temp_theta2 = Theta2;
temp_theta2(:,1) = zeros(num_labels,1);
sum2 = sum(sum(temp_theta2 .^ 2));
J = J + lambda/(2*m)*(sum1+sum2);

% backpropagation
Diff1 = zeros(size(Theta1,1), size(Theta1,2));
Diff2 = zeros(size(Theta2,1), size(Theta2,2));
for i=1:m
    % step 1
    a1 = X(i,:)'; % the i-th input
    z2 = Theta1*a1; % 25*1
    a2 = [1;sigmoid(z2)];
    %a2 = [1;a2]; % add ones to hidden layer
    z3 = Theta2*a2;
    a3 = sigmoid(z3);% h_theta(x) 
    %size(a3) % 10 * 1
    %size(new_y(i,:)')
    % step 2
    b3 = a3 - new_y(i,:)';
    % step 3
    % size(Theta2')
    % size(b3)
    % size(sigmoidGradient(z2))
    b2 = Theta2' * b3 .* sigmoidGradient([1;z2]);

    % remove 
    b2 = b2(2:end);
    Diff1 = Diff1 + b2*a1';
    Diff2 = Diff2 + b3*a2';
    
end 

% Theta1_grad = zeros(size(Theta1));
% Theta2_grad = zeros(size(Theta2));
Theta1_grad = Diff1/m;
Theta2_grad = Diff2/m;
for i=1:size(Theta1_grad,1)
    for j=2:size(Theta1_grad,2)
        Theta1_grad(i,j) = Theta1_grad(i,j) + lambda/m*Theta1(i,j);
    end
end
for i=1:size(Theta2_grad,1)
    for j=2:size(Theta2_grad,2)
        Theta2_grad(i,j) = Theta2_grad(i,j) + lambda/m*Theta2(i,j);
    end
end
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
