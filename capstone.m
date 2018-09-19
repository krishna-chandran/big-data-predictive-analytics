clc
load('C:\Users\telkapra\Downloads\Project4_Bearing\Dataset for Team\data_train.mat')
load('C:\Users\telkapra\Downloads\Project4_Bearing\Dataset for Team\data_train_labels.mat')
load('C:\Users\telkapra\Downloads\Project4_Bearing\Dataset for Team\data_test.mat') 

 

% filter data
data1=zeros(8191,2048);
for i=1:length(data_train)
    data1(:,i)= filter(data_train(i));
    
end

 data=[];
 for i=1:2048

% for i=1
 % data(:,i)= data1{1,i};
    data(:,i)=data1(:,i);
   inda_train=isoutlier(data1(:,i),'movmedian',100);

inda_train_healthy2=find(inda_train==1);

data_train_healthy2=[];
data_train_healthy2=data1(:,i);
data_train_healthy2(inda_train_healthy2)=[];


data(inda_train_healthy2,i)=mean(data_train_healthy2);
   
 
    
 end

%% Feature Extraction (Training)



Fs = 50000;   % Sampling rate
fr = 800/60;  % Spindle's rotating frequency (in Hz)
BPFO = 7.14*fr;  % outer defect freq.(in Hz)
BPFI = 9.88*fr;  % inner
BSF = 5.824*fr;  % roller

Data_mat=[];
for i=1:2048
    
%     figure(1)
%     plot(1:length(data),data,'b')

    mean_val=mean(data(:,i)); %Mean
    tempData = data(:,i) - repmat(mean_val,size(data(:,i),1),1);
    variance = var(tempData, 1);
    skew_val=skewness(data(:,i));
    kurtosis_val = kurtosis(data(:,i));
    rms_val = rms(data(:,i));
    peak_val = peak2peak(data(:,i));

     fft_data = fft(data(:,i));
     L=length(data(:,i));
     p2 = abs(fft_data/L);
     L_3=L/2+1;
     p1 = p2(1:L_3);
     p1(2:end-1) = 2*p1(2:end-1);
     f = 2560*(0:(L/2))/L;
     ind1 = find(f > BPFO-3 & f < BPFO+3 );
     ind2 = find(f > BPFI-3 & f < BPFI+3);
     ind3 = find(f > BSF-3 & f < BSF+3);
     ind4 = find(f > 80 & f < 85 );
     ind5 = find(f > 90 & f < 95 );
     plot(f,p1)

     Data_mat = [Data_mat ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3)),max(p2(ind4)),max(p2(ind5))];
 
end

%% Feature Extraction (Testing) 

% test_data=zeros(8191,512);
% for i=1:length(data_test)
%     test_data(:,i)= filter(data_test(i));
%     
% end

test_mat=[];
for i=1:512
    
    mean_val=mean(test_data(:,i)); %Mean
    tempData = test_data(:,i) - repmat(mean_val,size(test_data(:,i),1),1);
    variance = var(tempData, 1);
    skew_val=skewness(test_data(:,i));
    kurtosis_val = kurtosis(test_data(:,i));
    rms_val = rms(test_data(:,i));
    peak_val = peak2peak(test_data(:,i));

    fft_data = fft(test_data(:,i));
    L=length(test_data(:,i));
    p2 = abs(fft_data/L);
    L_3=L/2+1;
    p1 = p2(1:L_3);
    p1(2:end-1) = 2*p1(2:end-1);
    f = 2560*(0:(L/2))/L;
    ind1 = find(f > BPFO-3 & f < BPFO+3 );
     ind2 = find(f > BPFI-3 & f < BPFI+3);
     ind3 = find(f > BSF-3 & f < BSF+3);
    ind4 = find(f > 80 & f < 85 );
     ind5 = find(f > 90 & f < 95 );
     
    test_mat = [test_mat ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3)),max(p2(ind4)),max(p2(ind5))];
 
end

%% classes seperation
feature=zeros(8191,2048);

class=repmat({struct('class_data',{},'non_class_data',{})}, 8, 1);
after = [];
before = [];
for i = 1:8
    st.class_data=Data_mat(1+256*(i-1):256*i,:);
    if i ~= 8
    after = Data_mat(1+256*(i):256*8,:);
    end
    
    if i ~= 1
    before = Data_mat(1:256*(i-1),:);
    end
    st.non_class_data= [after;before];
    class{i} = st;
    after = [];
    before = [];
end

% 
%% Fisher Criterian
new_class=repmat({struct('class_data',{},'non_class_data',{},'top',{})}, 8, 1);
for f = 1:8
    Top=fisherFeatures(class{f}.class_data,class{f}.non_class_data);
    st1.class_data = class{f}.class_data(:,Top);
    st1.non_class_data = class{f}.non_class_data(:,Top);
    st1.top=Top;
    new_class{f}=st1;
    
    
end

%% Training SVM

svmStructure=repmat({struct('SupportVectors',{},'Alpha',{},'Bias',{},'KernelFunction',{},'KernelFunctionArgs',{},'GroupNames',{},'SupportVectorIndices',{},'ScaleData',{},'FigureHandles',{})}, 8, 1);
for i=1:8
    HealthyData=new_class{i}.class_data;
    DegradedData=new_class{i}.non_class_data;
    
    
    cross_count_healthy = length(HealthyData)*.75;
    cross_count_faulty = length(DegradedData)*.75;


    CrossTrainData = [HealthyData(1:cross_count_healthy,:);DegradedData(1:cross_count_faulty,:)];
    group = [linspace(1,1,cross_count_healthy)';linspace(0,0,cross_count_faulty)'];
    crossTestData_healthy = HealthyData(192:256,:);
    crossTestData_faulty = DegradedData(1344:1792,:);

    svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','rbf','showplot',false,'rbf_sigma',0.8);
    svmStructure{i} = svmStruct;
    health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',false);
    faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',false);
    
    predicted(i,:)=[health_test;faulty_test]';
    
    actual = [ones(8,65),zeros(8,449)];
%     figure
%     plotconfusion(actual,predicted);
    

    
end

train_result=ones(2048,1);
for u = 1:length(Data_mat)
    for p=1:8
        result1 = svmclassify(svmStructure{p},Data_mat(u,new_class{p}.top),'showplot',false);
        if(result1==1)
            train_result(u)=p;
            break
        end
        
    end
    
end



train_result1=train_result';
actual = [ones(1,256)*1,ones(1,256)*2,ones(1,256)*3,ones(1,256)*4,ones(1,256)*5,ones(1,256)*6,ones(1,256)*7,ones(1,256)*8];

act_res = zeros(2048,8);
fin_res = zeros(2048,8);
for e=1:2048
    fin_res(e,train_result1(e))=1;
    act_res(e,actual(e))=1;
end
  
plotconfusion(act_res',fin_res')


     
     
     
test_result=zeros(length(test_mat),1);
for n=1:length(test_mat)
    for m=1:8
        result = svmclassify(svmStructure{m},test_mat(n,new_class{m}.top),'showplot',false);
        if(result==1)
            test_result(n)=m;
            break
        end
        
    end

end


% Functions
function out= filter(data_train)
data = cell2mat(data_train(1));
% figure(1)
% plot(1:length(data),data,'b')
% figure(2)
% scatter(1:length(data),data)


 A = [ 0 1; 0 0 ];
 C = [ 0 1];
 P = [1 0; 0 1];
 Q = eye(2)*0.1;
R = eye(1)*5;

t = 1:length(data);


y_measure = data;

xhat = [1;data(1)];

est_xhat = xhat';
for i=2:length(data)-1
    dt = t(i) - t(i-1);
    for j=1:10
        xhat = xhat + (dt/10)*(A*xhat);
        P = P + (dt/10) *(A*P + P*A'+Q);
    end
    L = P *C'*(R+C*P*C')^-1;
    xhat = xhat + L*( y_measure(i)- xhat(2));
    P = (eye(2) - L*C)*P;
    est_xhat = vertcat(est_xhat,xhat');
end
out = est_xhat(:,2);
% figure(3)
% plot(1:length(est_xhat(:,2)),est_xhat(:,2),'b')
% figure(4)
% scatter(1:length(est_xhat(:,2)),est_xhat(:,2),'b')
end


function Top=fisherFeatures(GoodFeature,BadFeature)

    SelectNum = 2;
    for ii=1:size(GoodFeature,2) %for each column calculate
        if isempty(find(isnan(GoodFeature(:,ii)), 1)) && isempty(find(isnan(BadFeature(:,ii)), 1))
        Fisher(ii)=(mean(GoodFeature(:,ii))-mean(BadFeature(:,ii)))^2/(var(GoodFeature(:,ii))+var(BadFeature(:,ii)));  %fisher value
        else
            Fisher(ii)=0;
        end
    end

    [~, Order]=sort(Fisher,'descend');  %rank in descending order

    Top=Order(1:SelectNum);

end














