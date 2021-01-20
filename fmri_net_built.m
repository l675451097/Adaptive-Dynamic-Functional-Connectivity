function   [net] = fmri_net_built(fmri73,dti73,lambda)
% �������� DTI Weighted Guided ����Lasso����Ultra-OLS
% һϵ�е� lambda ������
% 
%  [err, idx] = ffols_gui(Y,P,threshold);
Net_fmri = zeros(90,90,73);
opts = [];
opts.init = 1;
opts.tFlag = 5;
opts.maxIter = 1000;
opts.nFlag = 0;
opts.rFlag = 1;

for i = 1:73  % �� i ������
    disp(['  ��',num2str(i),'������']);
    dti = dti73(:,:,i);
    fmri = fmri73(:,:,i);
    [fmri,~,~] = data_for_ultra(fmri);
    [Msize, Nsize] = size(fmri);
    for j = 1:Nsize  % �� j �� ROI
        w0 = dti(j,:);
        N0 = sum(w0~=0);
        deta = median(w0(w0~=0));
        if N0 == 0
            f0 = zeros(1,Nsize);
        else
            f0 = w0 ./ sum(w0);
        end
        f0(f0==1) = 3;
        Weight = exp(1-N0*f0);  %W_jk ������Ϊ0
        
        A = MatrixScale(fmri)./(ones(Msize,1)*Weight);
        A(:,j) = zeros(Msize,1);
        [Beta, funval, ~] = LeastR(A,MatrixScale(fmri(:,j)),lambda*(1-N0/(2*Nsize)),opts);
        ss = [w0',Beta];
        index = find(abs(Beta)> 1e-6);
        if any(index==j)
            msgbox('���ִ���ѡ���������ӣ���������','error','error');
            msg = 'Error happens !!';
            return;
        end
        Net_fmri(:,j,i) = Beta;
    end
end
net=Net_fmri;

function Mscaled = MatrixScale(MOrignal)
N = size(MOrignal,2);
Mscaled = zeros(size(MOrignal));
for i = 1:N
    temp = norm(MOrignal(:,i),2);
    if temp<eps
        continue;
    end
    Mscaled(:,i) = MOrignal(:,i)./temp;
end


