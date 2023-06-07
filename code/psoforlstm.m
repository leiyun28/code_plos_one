function [xm ,trace,result]=psoforlstm(p_train,t_train,p_test,t_test)

N=5;
M=10;
c1=1.5;
c2=1.5;
w=0.8;
D=4;

% Define the optimization frontier
xmin=[1    1  10 0.001];%Two pairs of hidden layer node training times and learning rate optimization
xmax=[200 200  100 0.01];%For example, the range of the first hidden layer node is 1-200


for i=1:N%Random initialization speed, random initialization position
    for j=1:D
        if j==D% % Hidden layer nodes and training times are integers Learning rate is floating point
            x(i,j)=(xmax(j)-xmin(j))*rand+xmin(j);
        else
            x(i,j)=round((xmax(j)-xmin(j))*rand+xmin(j));  %
        end
    end
    
    v(i,:)=rand(1,D);
end

%------First calculate the fitness of each particle, and initialize Pi and Pg----------------------
for i=1:N
    p(i)=fitness(x(i,:),p_train,t_train,p_test,t_test);
    y(i,:)=x(i,:);
    
end
[fg,index]=min(p);
pg = x(index,:);             %Pg is the global optimum

%------Enter the main loop and iterate sequentially according to the formula------------

for t=1:M
    
    for i=1:N
        v(i,:)=w*v(i,:)+c1*rand*(y(i,:)-x(i,:))+c2*rand*(pg-x(i,:));
        x(i,:)=x(i,:)+v(i,:);
        
        
        for j=1:D
            if j ~=D
                x(i,j)=round(x(i,j));
            end
            if x(i,j)>xmax(j) | x(i,j)<xmin(j)
                if j==D
                    x(i,j)=(xmax(j)-xmin(j))*rand+xmin(j);  %
                else
                    x(i,j)=round((xmax(j)-xmin(j))*rand+xmin(j));  %
                end
            end
        end
        temp=fitness(x(i,:),p_train,t_train,p_test,t_test);
        if temp<p(i)
            p(i)=temp;
            y(i,:)=x(i,:);
        end
        
        if p(i)<fg
            pg=y(i,:);
            fg=p(i);
        end
    end
    trace(t)=fg;
    result(t,:)=pg;
    t,fg,pg
end
xm = pg;