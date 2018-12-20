%% Clear and Close Figures
clear ; close all; clc;


%% Load Traning Data and visualize it
fprintf('Loading and Visualizing Data ...\n')
load('vowels.mat');
fprintf('Loading and Visualizing Data ...\n')
m = size(X, 1);
dimensions=size(X, 2);
% Print out 10  data points
show_points(X,i,y);
fprintf('Program paused. Press enter to continue.\n');
pause;

% Plot Data
plotData(X, y);%needs to be changed 'Ask Ahmed'
print -dpng 'myplot.png'
% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');
[X mu sigma] = featureNormalize(X);
% Add intercept term to X
X = [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta = normalEqn(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

prediction = 0; % You should change this
prediction=X*theta;

%calculation if the point is an outlier or not/we can set the threshold to be be 2.5 times the avg value
avg_value=mean(abs(y-prediction)*100);
threshold=1.5*avg_value;

check=prediction.*100;
check=check_threshold(check,threshold,m);
fprintf('Outlier detection matrix: \n');

%fprintf(' %f \n', check);
outliers_number=nnz(check);%non zero elements
actual_outlier_number=nnz(y); %actual outliers from the y matrix 
fprintf('Program paused. Press enter to continue.\n');
pause;

%print the outliers number
fprintf('outliers computed from the normal equations: \n');
fprintf(' %f \n', outliers_number);
fprintf('\n');
fprintf('outliers actually there: \n');
fprintf(' %f \n', actual_outlier_number);
fprintf('\n');
fprintf('Program paused. Press enter to continue.\n');
pause;

%miss rate/ hit rate calculation 
calculations(y,check,m,outliers_number,actual_outlier_number);
pause;
%%%%%%%%%%%%%%%%%%%%%%gradient descent method%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Program paused. Press enter to continue with gradient descent method.\n');
pause;
data = load('vowels.mat');
show_points(X,i,y);%show first 10 points

fprintf('Running gradient descent ...\n');
% Choose some alpha value
alpha = 0.1;
num_iters = 350;

theta = zeros(dimensions+1, 1);
[theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters);
%j_history teri purani cost hai

%% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');
fprintf('Program paused. Press enter to continue.\n');
pause;
print -dpng 'myplot1.png'

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

prediction = 0; % You should change this
prediction=X*theta;

%calculation if the point is an outlier or not/we can set the threshold to be be 2.5 times the avg value
avg_value=mean(abs(y-prediction)*100);
threshold=1.5*avg_value;
check=prediction.*100;
check=check_threshold(check,threshold,m);
fprintf('Outlier detection matrix: \n');

%fprintf(' %f \n', check);
outliers_number=nnz(check);%non zero elements
actual_outlier_number=nnz(y); %actual outliers from the y matrix 

%print the outliers number
fprintf('outliers computed from the gradient descent: \n');
fprintf(' %f \n', outliers_number);
fprintf('\n');
fprintf('outliers actually there: \n');
fprintf(' %f \n', actual_outlier_number);
fprintf('\n');

%miss rate/ hit rate calculation 
miss=calculations(y,check,m,outliers_number,actual_outlier_number);

%plotting the outlier on a plot
plot_outlier(check,m);
print -dpng 'myplot2.png'
data_map(X,check,miss);
title('Algorithm prediction');
print -djpg 'myplot3.jpg'
data_map(X,y,0);
title('Actual data value');
print -djpg 'myplot4.jpg'