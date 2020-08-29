function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
choices = [0.01, 0.03, .1, .3, 1, 3, 10, 30];
C =  1;
sigma = 0.3;

minError = 9e9;
bestIJ = [1,0.1];
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
%for i = choices
%  for j=choices
%    model= svmTrain(X, y, i, @(x1, x2) gaussianKernel(x1, x2, j));
%    predictions = svmPredict(model, Xval);
%    err = mean(double(predictions ~= yval));
%    if(err < minError)
%      minError=err;
%      bestIJ = [i, j];
%    endif
%  endfor
%endfor

C = bestIJ(1)
sigma = bestIJ(2)



% =========================================================================

end
