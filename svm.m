
clear all;close all;clc


%% Getting degraded training data
addpath(genpath('C:\Users\chandrkm\Downloads\Big Data\Big Data\SOM-Toolbox-master'))
jpegFiles = dir('C:\Users\chandrkm\Downloads\Big Data\Training\Training\Faulty\*.txt'); 
numfiles = length(jpegFiles);
mydata = cell(1, numfiles);

DegradedData = [];
for k = 1:numfiles 
	myfilename = sprintf('%s\\%s',jpegFiles(k).folder,jpegFiles(k).name);
    data=importfile(myfilename,6,38400);
   	data1=table2array(data(:,1));

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
        
    DegradedData = [DegradedData ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
 end
 
 

 
%% Getting healthy training data

jpegFiles = dir('C:\Users\chandrkm\Downloads\Big Data\Training\Training\Healthy\*.txt'); 
numfiles = length(jpegFiles);
mydata = cell(1, numfiles);

BaselineData = [];
for k = 1:numfiles 
	myfilename = sprintf('%s\\%s',jpegFiles(k).folder,jpegFiles(k).name);
    data=importfile(myfilename,6,38400);
   	data1=table2array(data(:,1));

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
    BaselineData = [BaselineData ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
 end 
 
%% Getting Test data

jpegFiles = dir('C:\Users\chandrkm\Downloads\Big Data\Testing\*.txt'); 
numfiles = length(jpegFiles);
mydata = cell(1, numfiles);

TestFeatureMatrix = [];
for k = 1:numfiles 
	myfilename = sprintf('%s\\%s',jpegFiles(k).folder,jpegFiles(k).name);
    data=importfile(myfilename,5,38400);
   	data1=table2array(data(:,1));

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
        
    TestFeatureMatrix = [TestFeatureMatrix ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
end 


% function [Top, Fisher] = FisherSelection(GoodFeature,BadFeature,SelectNum)
GoodFeature=BaselineData;
BadFeature=DegradedData;
SelectNum=3;

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
TrainData=[BaselineData(:,Top);DegradedData(:,Top)];
TestData=TestFeatureMatrix(:,Top);
group=[linspace(1,1,length(BaselineData))';linspace(0,0,length(DegradedData))'];

%% cross validation 

cross_count_healthy = length(BaselineData)*.75;
cross_count_faulty = length(DegradedData)*.75;
cross_test_healthy = length(BaselineData)- cross_count_healthy;
cross_test_faulty = length(DegradedData)- cross_count_faulty;

CrossTrainData = [BaselineData(1:cross_count_healthy,Top);DegradedData(1:cross_count_faulty,Top)];
group=[linspace(1,1,cross_count_healthy)';linspace(0,0,cross_count_faulty)'];
crossTestData_healthy = BaselineData(16:20,Top);
crossTestData_faulty = DegradedData(16:20,Top);




Methods = {'rbf', 'linear','polynomial','softmargin'};

for number = 1:4
    switch Methods{number}
        case 'rbf'
                % change rbf sigma value from 0.1 to 1, and observe dicision
                % boundary and number of support vectors
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','rbf','showplot',true,'rbf_sigma',0.1);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);

        case 'linear'
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','linear','showplot',true);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);

        case 'polynomial'
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','polynomial','showplot',true,'polyorder',5);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);

        case 'softmargin'
                % Try to change panelty value to observe decision boundary
                % shift
                PaneltyVersicolor =1;
                PaneltyVirginica  =1;
                svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','linear','showplot',true,'boxconstraint',[PaneltyVersicolor*ones(15,1);PaneltyVirginica*ones(15,1)]);
                health_test = svmclassify(svmStruct,crossTestData_healthy,'showplot',true);
                faulty_test = svmclassify(svmStruct,crossTestData_faulty,'showplot',true);
    end
    
    disp(Methods{number})
    disp(health_test)
    disp(faulty_test)
end


svmStruct = svmtrain(CrossTrainData,group,'Kernel_Function','polynomial','showplot',true,'polyorder',5);
health_test = svmclassify(svmStruct,TestData,'showplot',true);
stem(health_test)
