function difference_matrix=calculations(y,check,m,outliers_number,actual_outlier_number)
m=size(y,1);
difference_matrix=abs((y-check));%identifies the number of non zero terms 
miss=nnz(difference_matrix);
false_positives_number=size(find(y-check==-1),1);
false_negatives_number=size(find(y-check==1),1);
miss_rate=(miss/m)*100;
hit_rate=100-miss_rate;
fprintf('false_positives_number computed: \n');
fprintf(' %f \n', false_positives_number);
fprintf('false negatives computed: \n');
fprintf(' %f \n', false_negatives_number);
fprintf('miss rate computed: \n');
fprintf(' %f \n', miss_rate);
outlier_accuracy=(outliers_number/actual_outlier_number)*100;
fprintf('hit_rate computed: \n');
fprintf(' %f \n', hit_rate);
fprintf('outliers detection accuracy computed: \n');
fprintf(' %f \n', outlier_accuracy);

end