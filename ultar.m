function  [YY, PP, points_spline] = ultar(XX, order)
global N_regions;
[MM, NN] = size(XX);
if MM == 90 || MM == 116
    XX = XX(1:N_regions,:)';
    [MM, NN] = size(XX);
end
idx0 = (order+1:1:MM)';                                                                                                                                                                                        
idxx = repmat(idx0,1,order)-repmat(1:order,MM-order,1);
% YY = zeros(MM-order,NN);
% PP = zeros(MM-order,NN,order);
YY = zeros((3*(MM-order)-8),NN);
PP = zeros((3*(MM-order)-8),NN,order);
points_spline = 5;

%%  局部加权的方式
% M= [...
%   1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;              %1
%   1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;              %2
%    0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0;             %3
%    0 0 0 0  1 2 4 8 0 0 0 0 0 0 0 0;            %4
%    0 0 0 0 0 0 0 0 1 2 4 8 0 0 0 0;             %5
%    0 0 0 0 0 0 0 0 1 3 9 27 0 0 0 0;           %6
%    0 0 0 0 0 0 0 0 0 0 0 0 1 3 9 27;           %7
%    0 0 0 0 0 0 0 0 0 0 0 0 1 4 16 64;         %8
%    0 1 2 3 0 -1 -2 -3 0 0 0 0 0 0 0 0;        %9
%    0 0 0 0 0 1 4 12 0 -1 -4 -12 0 0 0 0;    %10
%    0 0 0 0 0 0 0 0 0 1 6 27 0 -1 -6 -27;    %11
%    0 0 1 3 0 0 -1 -3 0 0 0 0 0 0 0 0;         %12
%    0 0 0 0 0 0 1 6 0 0 -1 -6 0 0 0 0;         %13
%    0 0 0 0 0 0 0 0 0 0 1 9 0 0 -1 -9;         %14
%    0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0;            %15
%    0 0 0 0 0 0 0 0 0 0 0 0 0 1 8 48];         %16
% M2= [...
%   1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;              %1
%   1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0;              %2
%    0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0;             %3
%    0 0 0 0  1 2 4 8 0 0 0 0 0 0 0 0;            %4
%    0 0 0 0 0 0 0 0 1 2 4 8 0 0 0 0;             %5
%    0 0 0 0 0 0 0 0 1 3 9 27 0 0 0 0;           %6
%    0 0 0 0 0 0 0 0 0 0 0 0 1 3 9 27;           %7
%    0 0 0 0 0 0 0 0 0 0 0 0 1 4 16 64;         %8
%    0 1 2 3 0 -1 -2 -3 0 0 0 0 0 0 0 0;        %9
%    0 0 0 0 0 1 4 12 0 -1 -4 -12 0 0 0 0;    %10
%    0 0 0 0 0 0 0 0 0 1 6 27 0 -1 -6 -27;    %11
%    0 0 1 3 0 0 -1 -3 0 0 0 0 0 0 0 0;         %12
%    0 0 0 0 0 0 1 6 0 0 -1 -6 0 0 0 0;         %13
%    0 0 0 0 0 0 0 0 0 0 1 9 0 0 -1 -9;         %14
%    0 0 2 0 0 0 0 0 0 0 0 0 0 0 0 0;            %15
%    0 0 0 0 0 0 0 0 0 0 0 0 0 0 2 24];         %16
% A = inv(M);
% B = inv(M2);
% for i=1:NN
%     temp = XX(:,i);
%     tYY = temp(idx0);
%     YY(:,i) = reinforce_self_fai(A,B,tYY); 
%     for j= 1:order
%         tPP = temp(idxx(:,j));
%         PP(:,i,j) = reinforce_self_fai(A,B,tPP);
%     end
% end
%%
%%  非局部加权的方式
for i=1:NN
    temp = XX(:,i);
    tYY = temp(idx0);
    YY(:,i) = reinforce(tYY); 
    for j= 1:order
        tPP = temp(idxx(:,j));
        PP(:,i,j) = reinforce(tPP);
    end
end
PP = reshape(PP,(3*(MM-order)-8),NN*order);
%%



function  XS = reinforce(X)
% 扩展点代码，使用3次样条函数，1、2阶导数
% 一阶导 [0 1 0 1 0]
% 二阶导 [0 1 -2 1 0]
y = X(:);
k_1 = [0;1;0;-1;0]; k_1 = k_1/norm(k_1,2);  
k_2 = [0;1;-2;1;0]; k_2 = k_2/norm(k_2,2);
Nlen = length(y);    
Nk = length(k_1);
ind_Nk1 = (1:Nlen-Nk+1)';
ind_Nk = repmat(ind_Nk1,1,Nk)+repmat(0:Nk-1,Nlen-Nk+1,1);
PP_y = y(ind_Nk);
y_1 = PP_y*k_1;  %扩展后的 y 的 1阶弱微分部分
y_2 = PP_y*k_2;  %扩展后的 y 的2阶弱微分部分
y = y-mean(y);
y = y./norm(y);
y_1 = y_1 - mean(y_1);
y_2 = y_2 - mean(y_2);
y_1 = y_1./norm(y_1);
y_2 = y_2./norm(y_2);

XS = cat(1,y,y_1,y_2);  %扩展后的总的 y 点


function  XS = reinforce_self_fai(A,B,X)
% 扩展点代码，使用3次样条函数，1、2阶导数
y = X(:);
% k_1 = [0;1;0;-1;0]; k_1 = k_1/norm(k_1,2);  
% k_2 = [0;1;-2;1;0]; k_2 = k_2/norm(k_2,2);
% Nk = length(k_1);
Nlen = length(y);    
Nk = 5;
ind_Nk1 = (1:Nlen-Nk+1)';
ind_Nk = repmat(ind_Nk1,1,Nk)+repmat(0:Nk-1,Nlen-Nk+1,1);
PP_y = y(ind_Nk);
[k_1, k_2] = Self_Bspline_3_order(A,B,PP_y);
y_1 = PP_y*k_1;
y_2 = PP_y*k_2;
XS = cat(1,y,diag(y_1),diag(y_2));  %扩展后的总的 y 点









