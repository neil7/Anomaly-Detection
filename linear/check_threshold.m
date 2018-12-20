function check = check_threshold(check,threshold,m);
%% converts the check matrix to a outlier detection matrix if (check>threshold)
 for i=1:m
  if(check(i)>threshold)
  check(i)=1;
  else
  check(i)=0;
  end
end