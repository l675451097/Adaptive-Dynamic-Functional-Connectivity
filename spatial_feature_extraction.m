function  [msg] = spatial_feature_extraction(  )

pd = pwd;
mkdir('73samples_Feature_all');
cd('.\高阶网络结果\');
nams = dir('*.mat');
for ii = 1:length(nams)
    
    str_name = nams(ii).name;
    load(str_name);
    nets = high_net(1:187,1:187,:); %#ok

    feas = zeros(73,187,4);
    for i = 1:73
        feas(i,:,1) = clustering_coef_wd(nets(:,:,i)); %
        feas(i,:,2) = betweenness_bin(nets(:,:,i));
        [feas(i,:,3), feas(i,:,4)] = built_net_degree(nets(:,:,i));
    end
    save([pd,'\73samples_Feature_all\Fea_',str_name(11:end)],'feas');
end
cd(pd);
msg = 'done, ok!';

function BC=betweenness_bin(G)
%BETWEENNESS_BIN    Node betweenness centrality
%
%   BC = betweenness_bin(A);
%
%   Node betweenness centrality is the fraction of all shortest paths in 
%   the network that contain a given node. Nodes with high values of 
%   betweenness centrality participate in a large number of shortest paths.
%
%   Input:      A,      binary (directed/undirected) connection matrix.
%
%   Output:     BC,     node betweenness centrality vector.
%
%   Note: Betweenness centrality may be normalised to [0,1] via BC/[(N-1)(N-2)]
%
%   Reference: Kintali (2008) arXiv:0809.1906v2 [cs.DS]
%              (generalised to directed and disconnected graphs)
%
%
%   Mika Rubinov, UNSW, 2007-2010


n=length(G);                %number of nodes
I=eye(n)~=0;                %logical identity matrix
d=1;                     	%path length
NPd=G;                      %number of paths of length |d|
NSPd=NPd;                  	%number of shortest paths of length |d|
NSP=NSPd; NSP(I)=1;        	%number of shortest paths of any length
L=NSPd; L(I)=1;           	%length of shortest paths

%calculate NSP and L
while find(NSPd,1)
    d=d+1;
    NPd=NPd*G;
    NSPd=NPd.*(L==0);
    NSP=NSP+NSPd;
    L=L+d.*(NSPd~=0);
end
L(~L)=inf; L(I)=0;          %L for disconnected vertices is inf
NSP(~NSP)=1;                %NSP for disconnected vertices is 1

Gt=G.';
DP=zeros(n);            	%vertex on vertex dependency
diam=d-1;                  	%graph diameter

%calculate DP
for d=diam:-1:2
    DPd1=(((L==d).*(1+DP)./NSP)*Gt).*((L==(d-1)).*NSP);
    DP=DP + DPd1;       %DPd1: dependencies on vertices |d-1| from source
end

BC=sum(DP,1);               %compute betweenness

% 自定义函数，特征构建子函数
function [fea_out, fea_in] = built_net_degree(temp)   %出度/入度
  temp = (abs(temp)>0.000001);
  temp = temp - diag(diag(temp));
  fea_in = sum(temp,1)/sum(sum(temp,1));
  fea_out = sum(temp,2)'/sum(sum(temp,2));


  
function C=clustering_coef_wd(W)                % 计算有向图的集聚系数
%CLUSTERING_COEF_WD     Clustering coefficient
%
%   C = clustering_coef_wd(W);
%
%   The weighted clustering coefficient is the average "intensity" of 
%   triangles around a node.
%
%   Input:      W,      weighted directed connection matrix
%
%   Output:     C,      clustering coefficient vector
%
%   Reference: Fagiolo (2007) Phys Rev E 76:027307.
%
%
%   Mika Rubinov, UNSW, 2007-2010


%Methodological note (also see clustering_coef_bd)
%The weighted modification is as follows:
%- The numerator: adjacency matrix is replaced with weights matrix ^ 1/3
%- The denominator: no changes from the binary version
%
%The above reduces to symmetric and/or binary versions of the clustering 
%coefficient for respective graphs.

A=W~=0;                     %adjacency matrix
S=W.^(1/3)+(W.').^(1/3);	%symmetrized weights matrix ^1/3
K=sum(A+A.',2);            	%total degree (in + out)
cyc3=diag(S^3)/2;           %number of 3-cycles (ie. directed triangles)
K(cyc3==0)=inf;             %if no 3-cycles exist, make C=0 (via K=inf)
CYC3=K.*(K-1)-2*diag(A^2);	%number of all possible 3-cycles
C=cyc3./CYC3;               %clustering coefficient
C = C';


