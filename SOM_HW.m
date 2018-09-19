
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
     
%      plot(f,p1)
        
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
SelectNum=1;

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

sDIris = som_data_struct(TrainData,'name','FisherIris',...
			  'comp_names',{'1'});
          

sDIris = som_label (sDIris,'add',[1:20],'healthy');
sDIris = som_label (sDIris,'add',[21:40],'faulty');

sM = som_make(sDIris);

sMap = som_autolabel(sM,sDIris,'vote');


som_show(sMap);
%% Calculate the MQE values for the testing data set
S=size(TestData);
S=S(1);
for ii=1:S
    [qe te]=som_quality(sM,TestData(ii,:)); % calculate MQE value for each sample
    MQEt(ii)=qe;
end

MQEtn=(1-(MQEt)./(max(MQEt))); % normalize MQE
MQEtn=MQEtn';

%% Plot the calculated MQE values 
% observe the difference between normal condition and faluty conditions
plot(MQEtn,'-*');
xlabel('Data file No.');
 ylabel('Confidence value (MQE)');
 title('Health Assessment Plot');

