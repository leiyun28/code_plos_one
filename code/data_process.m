function [in,out]=data_process(data,num)
% Take 1-num as input and num+1 as output
n=length(data)-num;
for i=1:n
    x(i,:)=data(i:i+num);
end
in=x(:,1:end-1);
out=x(:,end);