function [msg]=highorder_net_built()
load('fmri_dti_73.mat')
fmri=zeros(180,90,73);
dti=zeros(90,90,73);
for i=1:73
     fmri(:,:,i)=fmri_dti_73{i,2}(:,1:90);
     dti (:,:,i)= fmri_dti_73{i,5};
end
mkdir('rls网络结果');
mkdir('cts结果');
mkdir('cts1结果');
for lamda=[0.01:0.01:0.3]
    load(['低阶网络结果\low_net_lamda_',num2str(lamda),'.mat'],'low_net');
    n=0.95;
    rlsnet=zeros(90,90,180,73);
    for i=1:73
        for j=1:90
            index = find(low_net(j,:,i)>0.01);
            [~,pa]=FFRLS(fmri(:,j,i),fmri(:,index,i),n);
            for k=1:180
                rlsnet(j,index,k,i)=pa(:,k)';
            end
        end
    end
    save(['rls网络结果\rlsnet_lamda_',num2str(lamda),'.mat'],'rlsnet');
    cts=zeros(180,8010,73);
    k=0;
    for i=1:90
        for j=setdiff(1:90,i)
            k=k+1;
            for z=1:73
                cts(:,k,z)=squeeze(rlsnet(i,j,:,z));
            end
        end
    end
    save(['cts结果\cts_lamda_',num2str(lamda),'.mat'],'cts');
    [xx,y,z]=size(cts);
    cts1=zeros(xx*z,y);
    for j=1:z
        cts1((xx*j-xx+1):(xx*j),:)=cts(:,:,j);
    end
    save(['cts1结果\cts1_lamda_',num2str(lamda),'.mat'],'cts1');
end

mean1=zeros(180,187,73);
k=0;
for i=1:8010
    if length(find(cts1(:,i)==0))<180*73/2
         k=k+1;
         mean1(:,k,:)=cts(:,i,:);
    end
end
save('mean1','mean1');

mkdir('高阶网络结果');
for lamda=0.01:0.01:0.3
    [high_net] = net_built_ultar_lasso_OLS(mean1,0,lamda,0.001);
    save(['高阶网络结果\Net_order0_lamda_',num2str(lamda),'.mat'],'high_net');
end
msg = 'done';