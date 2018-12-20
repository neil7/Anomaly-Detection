function plot_outlier(y,m)
figure;
x=0;
for i=1:m
if(y(i)==1)
  plot(x,y(i), 'rx','MarkerSize', 10);
  x=x+0.2;
  hold on;  
elseif(y(i)==0)
  plot(x,y(i), 'gx','MarkerSize', 10);
  x=x+0.2;
  hold on;
end;
end;  
ylabel('Outlier detection');
xlabel('Total data points');
title('Outlier mapping')