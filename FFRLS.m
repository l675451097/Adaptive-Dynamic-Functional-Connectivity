%FFRLS
%已知输出Y（M*1），输入phi（M*N）
%n：遗忘因子 0<n<=1，n需接近1，一般不小于0.9
function [th,pa]=FFRLS(Y,phi,n)

% [phi T]=gp(Y,u,py,pu,d);

[M,N]=size(phi);
%初值
P=10^4*eye(N);
th_1=zeros(N,1);

for i=2:M
    fy=phi(i,:)';
    y=Y(i,:);
    K=(P*fy)/(n+fy'*P*fy);
    th{i}=th_1+K*(y-fy'*th_1);
    P=(1.0/n)*(eye(N)-K*fy')*P;
    
    th_1=th{i};
end

pa=zeros(N,M);
for i=1:N
    for j=2:M
        pa(i,j)=th{j}(i,:);
    end
end

% for i=2
%     for j=1:N
%         if (pa(i,j)>1)|(pa(i,j)<-1)
%             pa(i,j)=0;
%         end
%     end
% end
% for i=4
%     for j=1:N
%         if (pa(i,j)>1)|(pa(i,j)<-1)
%             pa(i,j)=0;
%         end
%     end
% end

% Ye1=zeros(N,1);
% Ye2=zeros(N,1);
% % Ye1(2,:)=pa(1,2)*Y(1,:)+pa(3,2)*u(1);
% % Ye2(2,:)=pa(1,2)*Y(1,:);
% for i=2:N
%     Ye1(i,:)=phi(i,:)*th{i};
% %     Ye2(i,:)=phi(i,1)*th{i}(1,:)+phi(i,2)*th{i}(2,:);
% % Ye1(i,:)=pa(1,i)*Y(i-1,:)+pa(2,i)*Y(i-2,:)+pa(3,i)*u(i-1)+pa(4,i)*u(i-2);
% % Ye2(i,:)=pa(1,i)*Y(i-1,:)+pa(2,i)*Y(i-2,:);
% end
% for i=2:N
%     Ye1(i,:)=pa(1,i)*Y(i-1,:)+pa(2,i)*u(i-1)+pa(3,i)*(Y(i-1,:))^2+pa(4,i)*(u(i-1))^2;
%     Ye2(i,:)=pa(1,i)*Y(i-1,:)+pa(3,i)*(Y(i-1,:))^2;
% end

end

%generatedata3 test ok
% clear all
% load('testdata3_u.mat')
% load('testdata3_Y.mat')
% load('testdata3_Y.mat')
% [P]=generatep(Y,u,2,4,0);
% phi=[P(:,1) P(:,2) P(:,5) P(:,6)];
% [th,m]=FFRLS(Y,phi,0.98);
% figure;plot(1:1000,m(1,:));xlabel('t');ylabel('a1');
% figure;plot(1:1000,m(2,:));xlabel('t');ylabel('a2');
% figure;plot(1:1000,m(3,:));xlabel('t');ylabel('b1');
% figure;plot(1:1000,m(4,:));xlabel('t');ylabel('b2');




















