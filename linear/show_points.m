function show_points(X, i,y)
% Print out 10  data points
fprintf('First 10 examples from the dataset: \n');
i=1;
while i<=10
  disp(X(i,:)),disp('y='),disp(y(i,:));
  fprintf('\n'); 
  i=i+1;
end;