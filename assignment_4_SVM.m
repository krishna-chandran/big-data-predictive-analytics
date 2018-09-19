
clear all;close all;clc


%% Getting degraded training data

cd 'D:\Alive\IMS\Big data\Homework 4-5 group work\feb 25\Big Data\Big Data\Big data\Training\Training\Faulty\Unbalance 1';
d=dir('*.txt');
 
numfiles = length(d);

DegradedData1 = [];
for k = 1:numfiles 

filename = d(k).name;
fidn = fopen(filename);
C = textscan(fidn, '%s');
fclose(fidn);
 
S0= C{1, 1}(12:end);
S1 = sprintf('%s*', S0{:});
data1 = sscanf(S1, '%f*');
%     data=importfile(myfilename,6,38400);
%    	data1=table2array(data(:,1));

    mean_val=mean(data1); %Mean
    tempData = data1 - repmat(mean_val,size(data1,1),1);
    variance = var(tempData, 1);
    skew_val=skewness(data1);
    kurtosis_val = kurtosis(data1);
    rms_val = rms(data1);
    peak_val = peak2peak(data1);

     fft_data = fft(data1);
     L=length(data1);
     p2 = abs(fft_data/L);
     L_3=L/2+1;
     p1 = p2(1:L_3);
     p1(2:end-1) = 2*p1(2:end-1);
     f = 2560*(0:(L/2))/L;
     ind1 = find(f > 20 & f < 22);
     ind2 = find(f > 40 & f < 50);
     ind3 = find(f > 60 & f < 70);
     
   % plot(f,p1)
        
    DegradedData1(k,:)= [mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
 end
 
 cd 'D:\Alive\IMS\Big data\Homework 4-5 group work\feb 25\Big Data\Big Data\Big data\Training\Training\Faulty\Unbalance 2';
d=dir('*.txt');
 
numfiles = length(d);

DegradedData2 = [];
for k = 1:numfiles 

filename = d(k).name;
fidn = fopen(filename);
C = textscan(fidn, '%s');
fclose(fidn);
 
S0= C{1, 1}(12:end);
S1 = sprintf('%s*', S0{:});
data1 = sscanf(S1, '%f*');
%     data=importfile(myfilename,6,38400);
%    	data1=table2array(data(:,1));

    mean_val=mean(data1); %Mean
    tempData = data1 - repmat(mean_val,size(data1,1),1);
    variance = var(tempData, 1);
    skew_val=skewness(data1);
    kurtosis_val = kurtosis(data1);
    rms_val = rms(data1);
    peak_val = peak2peak(data1);

     fft_data = fft(data1);
     L=length(data1);
     p2 = abs(fft_data/L);
     L_3=L/2+1;
     p1 = p2(1:L_3);
     p1(2:end-1) = 2*p1(2:end-1);
     f = 2560*(0:(L/2))/L;
     ind1 = find(f > 20 & f < 22);
     ind2 = find(f > 40 & f < 50);
     ind3 = find(f > 60 & f < 70);
     
   % plot(f,p1)
        
    DegradedData2(k,:)= [mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
 end
 

 DegradedData=[DegradedData1;DegradedData2];
%% Getting healthy training data

cd 'D:\Alive\IMS\Big data\Homework 4-5 group work\feb 25\Big Data\Big Data\Big data\Training\Training\Healthy';
d=dir('*.txt');
numfiles = length(d);

BaselineData = [];
for k = 1:numfiles 
	filename = d(k).name;
fidn = fopen(filename);
C = textscan(fidn, '%s');
fclose(fidn);
 
S0= C{1, 1}(12:end);
S1 = sprintf('%s*', S0{:});
data1 = sscanf(S1, '%f*');

    mean_val=mean(data1); %Mean
    tempData = data1 - repmat(mean_val,size(data1,1),1);
    variance = var(tempData, 1);
    skew_val=skewness(data1);
    kurtosis_val = kurtosis(data1);
    rms_val = rms(data1);
    peak_val = peak2peak(data1);

     fft_data = fft(data1);
     L=length(data1);
     p2 = abs(fft_data/L);
     L_3=L/2+1;
     p1 = p2(1:L_3);
     p1(2:end-1) = 2*p1(2:end-1);
     f = 2560*(0:(L/2))/L;
   ind1 = find(f > 20 & f < 22);
     ind2 = find(f > 40 & f < 50);
     ind3 = find(f > 60 & f < 70);
    BaselineData(k,:) = [ mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
 end 
 
%% Getting Test data

cd 'D:\Alive\IMS\Big data\Homework 4-5 group work\feb 25\Big Data\Big Data\Big data\Testing';
d=dir('*.txt');
numfiles = length(d);


TestFeatureMatrix = [];
for k = 1:numfiles 
	filename = d(k).name;
fidn = fopen(filename);
C = textscan(fidn, '%s');
fclose(fidn);
 
S0= C{1, 1}(12:end);
S1 = sprintf('%s*', S0{:});
data1 = sscanf(S1, '%f*');

    mean_val=mean(data1); %Mean
    tempData = data1 - repmat(mean_val,size(data1,1),1);
    variance = var(tempData, 1);
    skew_val=skewness(data1);
    kurtosis_val = kurtosis(data1);
    rms_val = rms(data1);
    peak_val = peak2peak(data1);

     fft_data = fft(data1);
     L=length(data1);
     p2 = abs(fft_data/L);
     L_3=L/2+1;
     p1 = p2(1:L_3);
     p1(2:end-1) = 2*p1(2:end-1);
     f = 2560*(0:(L/2))/L;
     ind1 = find(f > 20 & f < 22);
     ind2 = find(f > 40 & f < 50);
     ind3 = find(f > 60 & f < 70);
     %plot(f,p1)
        
    TestFeatureMatrix(k,:) = [ mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
end 


% function [Top, Fisher] = FisherSelection(GoodFeature,BadFeature,SelectNum)
GoodFeature=BaselineData;
BadFeature=DegradedData;
SelectNum=2;

%inputs - Good Feature (Features from baseline data)
       %- Bad Featur (feature from degraded/faulty component/system)
       % SelectNum - how many to select (top5, top10, etc)

%Outputs - Top Column Numbers, Fisher Values

for ii=1:size(GoodFeature,2) %for each column calculate
    if isempty(find(isnan(GoodFeature(:,ii)), 1)) && isempty(find(isnan(BadFeature(:,ii)), 1))
    Fisher(ii)=(mean(GoodFeature(:,ii))-mean(BadFeature(:,ii)))^2/(var(GoodFeature(:,ii))+var(BadFeature(:,ii)));  %fisher value
    else
        Fisher(ii)=0;
    end
end

%rank Fisher
[FisherRanked, Order]=sort(Fisher,'descend');  %rank in descending order

Top=Order(1:SelectNum);  %select the top so many
% end
%% 
TrainDataF=[BaselineData(:,Top);DegradedData(:,Top)];
TestData=TestFeatureMatrix(:,Top);
group=[linspace(0,0,length(BaselineData))';linspace(1,1,length(DegradedData1))';linspace(2,2,length(DegradedData2))'];

%% cross validation 
% 
 cross_count_healthy = length(BaselineData)*.75;
% cross_count_healthy = 10;
cross_count_faulty = length(DegradedData1)*.75;
% cross_test_healthy = length(BaselineData)- cross_count_healthy;
% cross_test_faulty = length(DegradedData)- cross_count_faulty;

CrossTrainData = [BaselineData(1:cross_count_healthy,Top);DegradedData1(1:cross_count_faulty,Top)];
group=[linspace(1,1,cross_count_healthy)';linspace(0,0,cross_count_faulty)'];
crossTestData_healthy = BaselineData(16:20,Top);
crossTestData_faulty = DegradedData1(16:20,Top);




Methods = {'rbf', 'linear','polynomial','softmargin'};

for number = 1:4
    switch Methods{number}
        case 'rbf'
             figure
                % change rbf sigma value from 0.1 to 1, and observe dicision
                % boundary and number of support vectors
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','rbf','showplot',true,'rbf_sigma',0.1);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                hold on
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);
title('Classification method rbf')
        case 'linear'
             figure
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','linear','showplot',true);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);
title('Classification method- linear')
        case 'polynomial'
             figure
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','polynomial','showplot',true,'polyorder',5);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);
title('Classification method- polynomial') 
        case 'mlp'
                % Try to change panelty value to observe decision boundary
                % shift
                figure
                PaneltyVersicolor =1;
                PaneltyVirginica  =1;
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','mlp','showplot',true,'boxconstraint',[PaneltyVersicolor*ones(15,1);PaneltyVirginica*ones(15,1)]);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);
               title('Classification method- mlp') 
    end
    
%     figure(number)
%     disp(Methods{number})
%     hold on
%     disp(health_test)
%     hold on
%     disp(faulty_test)
 
end

%%
Trainlabels={'Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2'}';
Testlabels={'Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 1','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2','Unbalance Level 2'}';
% Unique class labels
uniqueclasses = unique(Trainlabels);
% Training class labels
train_classnum = zeros(length(Trainlabels),1);
for ii = 1:length(Trainlabels)
    train_classnum(ii) = find(strcmp(uniqueclasses,Trainlabels{ii,1})==1);
end
% Testing class labels
 uniqueclasses = unique(Testlabels);

test_classnum = zeros(length(Testlabels),1);
for ii = 1:length(Testlabels)
    test_classnum(ii) = find(strcmp(uniqueclasses,Testlabels{ii,1})==1);
end
% number of testing data sample
NTest = size(TestData, 1);


    %% Health diagnosis using MATLAB built-in SVM
    %% Training
     % Add the path to the MATLAB built-in svmtrain function to the top of
     % MATLAB search paths to ensure that the built-in function will be
     % used when call 'svmtrain' function in the following codes.
     % Notice: please change the following path according to actural
     % toolbox's location.
    addpath(genpath('C:\Program Files\MATLAB\R2015a\toolbox\stats\stats'));
    
    All_svmStruct = cell(2,3);
 
    success_rate = [];
    ctr = 1;
    for ii = 1:2
        for jj = ii+1:3
            index1 = find(train_classnum==ii==1);
            index2 = find(train_classnum==jj==1);
            training_index = [index1; index2];
            training_target = [ones(length(index1),1); zeros(length(index2),1)];
            All_svmStruct{ii,jj} = svmtrain(TrainDataF(training_index,:),training_target,...
                'Kernel_Function','polynomial','showplot',false);
            classes = svmclassify(All_svmStruct{ii,jj},TrainDataF(training_index,:));
            success_rate(ctr,1) = (length(training_index) - sum(xor(training_target,classes)))/length(training_index);
            ctr = ctr+1;
        end
    end
    %% Testing
    test_class = zeros(NTest,1);
    for ii = 1:NTest
        tempclass = 1;
        for jj = 1:2
            sample_class = svmclassify(All_svmStruct{tempclass,jj+1},TestData(ii,:));
            if ~sample_class
                tempclass = jj+1;
            end
        end
        test_class(ii) = tempclass;
    end
    
    
%     plot(sample_class,'*')
%     plot(test_class,'*r')
    
    %% Calculate successful rate
    tempx = test_classnum - test_class;
    test_success_rate = (length(test_classnum) - length(find(tempx == 0 == 0)))/length(test_classnum);
    %% Calculate the confusion matrix
     % Calculation of Confusion Matrix based on variables 'test_classnum'
     % and 'test_class'. Function 'confusionmat' will be used.
     
%      cd('C:\Users\IMS_Pin\Dropbox\Training\7-FaultDiagnosis_SVM')
     % ================= Your Code Here ====================
     [C,class] = confusionmat(test_classnum,test_class);
     opt = confMatPlot('defaultOpt');
     opt.className = uniqueclasses;
     opt.mode = 'both';
     
     figure; confMatPlot(C, opt);
     
