function data_map(X, y,miss)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.

% Find Indices of Positive and Negative Examples
[rows_1 ,coloumns_1] = find(y==1);%store the index of matrix of 1 in pos
[rows_2 ,coloumns_2] = find(y == 0);%store the index of matrix of 0 in neg
[rows_3 ,coloumns_3] = find(miss==1);
% Plot Examples
plot(X(rows_1, 2), X(rows_1, 3), 'k+','LineWidth', 2, ...
'MarkerSize', 7);%denotes an outlier point
plot(X(rows_2, 2), X(rows_2, 3), 'ko', 'MarkerFaceColor', 'y', ...
'MarkerSize', 7);%denotes a normal point
plot(X(rows_3,2),  X(rows_3, 3), 'r+','LineWidth', 2, ...
'MarkerSize', 7);%denotes a false positive point
xlabel('feature1');
ylabel('feature2');
legend('Outlier points', 'Data points','false positives');
% =========================================================================
hold off;

end