function [X_train, X_test, y_train, y_test]=divide(X,y)
division_ratio=70;
fprintf('divison ratio: \n');
fprintf(' %f \n', division_ratio);
division_number=(division_ratio/100);%computes the total number of rows in train data 
%%%% 1 to division row data in train and division row to end data in test
[rows_1 ,coloumns_1] = find(y==1);%store the index of matrix of 1 in pos
[rows_2 ,coloumns_2] = find(y == 0);%store the index of matrix of 0 in ne
outlier_number=size(rows_1,1);
nonoutlier_number=size(rows_2,1);
count_o=round(division_number*outlier_number);%count of outier number in train data
fprintf('The split divison number you set for outlier: \n');
fprintf(' %f \n', count_o);
count_d=round(division_number*nonoutlier_number)%count of data number in train
fprintf('The split divison number you set for data: \n');
fprintf(' %f \n', count_d);
ma=X((rows_1(1:count_o)),:);%70 of the coutlier data 
pa=X((rows_2(1:count_d)),:);%70 of the normal data
X_train=[ma;pa];
ma_1=X(rows_1(count_o+1:end),:);%remaining 30 of the outlier data in train
ma_2=X(rows_2(count_d+1:end),:);%remaining 30 of the normal data
X_test=[ma_1;ma_2];
y_train=[y(rows_1(1:count_o),:);y(rows_2(1:count_d))];
y_test=[y(rows_1(count_o+1:end),:);y(rows_2(count_d+1:end))];
end