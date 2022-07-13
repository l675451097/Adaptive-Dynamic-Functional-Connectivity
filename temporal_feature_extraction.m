mkdir('73samples_TFeature_all');
mkdir('mean1结果');
mkdir('187溯源结果');
pd = pwd;
cd('.\cts1结果\');
nams = dir('*.mat');
for ii = 1:length(nams)
    str_name = nams(ii).name;
    load(str_name);
    load([pd,'\cts结果\cts_',str_name(6:end)]);
    k=0;
    for i=1:8010
        if length(find(cts1(:,i)==0))<180*73/2
            k=k+1;
        end
    end
    mean1=zeros(180,k,73);
    suyuan=zeros(180,2);
    k=0;
    for i=1:8010
        if length(find(cts1(:,i)==0))<180*73/2
            k=k+1;
            mean1(:,k,:)=cts(:,i,:);
            roi1=floor((i-1)/89)+1;
            suyuan(k,1)=roi1;
            roi2=i-(roi1-1)*89;
            if roi2>=roi1
                roi2=roi2+1;
            end
            suyuan(k,2)=roi2;
        end
    end
    save([pd,'\mean1结果\mean1',str_name(6:end)],'mean1');
    save([pd,'\187溯源结果\suyuan',str_name(6:end)],'suyuan');
    Tfeas=zeros(73,k,2);
    for i=1:73
        for j=1:k
            Tfeas(i,j,1)=std(mean1(:,j,i),1);
            Tfeas(i,j,2) = rms(mean1(:,j,i));
        end
    end
    save([pd,'\73samples_TFeature_all\TFea_',str_name(6:end)],'Tfeas');
end
