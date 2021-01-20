function  [net] = net_built_ultar_lasso_OLS1(P,order,lamda,threshold)
[~,n,t]=size(P);
net=zeros(n,n,t);
Ys = [];
Ps = [];
for i = 1:t
    [YY, PP,~] = data_for_ultra(P(:,:,i),order);
    Ys = cat(1,Ys,YY);
    Ps = cat(1,Ps,PP);
end
YY=Ys;
PP=Ps;

for j=1:n
    opts = [];
    opts.q = 2;
    opts.init = 2;
    opts.tFlag = 5;
    opts.maxIter = 1000;
    opts.nFlag = 0;
    opts.rFlag = 1;
    opts.ind = [];
    [MM, Nsize] = size(YY);
    Ys = YY(:,j);
    Ps = [YY,PP];
    Ps(:,j) = zeros(MM,1);
    opts.ind =0:MM/t:MM;
    [x1, ~, ~] = mtLeastR(Ps,Ys,lamda,opts);
    index = find(abs(x1(:,1))>0.0000001);
    ind_end = opts.ind;
    for i = 1:t
        Y = Ys((ind_end(i)+1):ind_end(i+1));
        if sum(abs(Y))>0 && (~isempty(index))
            P = Ps((ind_end(i)+1):ind_end(i+1),index);
            [err, idx] = ffols_gui(Y,P,threshold);
            idx_true = index(idx+1);
            theta = (P(:,(idx+1))'*P(:,(idx+1)))\P(:,(idx+1))'*Y;
            [net(j,:,i)] = net_out(idx_true,err,Nsize,j);
            infos.sample(i).err = err;
            infos.sample(i).theta = theta;
            infos.sample(i).idx = idx_true;
            infos.sample(i).rho(j)=lamda;
        end
    end
    disp(['建网络已完成',num2str(j/n*100),'%']);
end
net(isnan(net)) = 0;
end
%%
function [net] = net_out(idx,err,Nsize,ind_i)
if ~isempty(find((idx==(ind_i)),1))
    disp('发生了自连接','error');
    error('发生了自连接');
end
net = zeros(1,Nsize);
ind = mod(idx,Nsize);
ind(ind==0) = Nsize;
for z = 1:Nsize
    tind = find(ind==z);
    if isempty(tind)
        net(1,z) = 0;
    else
        net(1,z) = sum(err(tind));  % err相加
    end
end
end %对应function net_out