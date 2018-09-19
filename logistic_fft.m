
clear all;close all


%% Getting degraded training data

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
     ind = find(f > 20 & f < 22);
     ind2 = find(f > 40 & f < 50);
     ind3 = find(f > 60 & f < 70);
     
%      plot(f,p1)
        
    DegradedData = [DegradedData ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind)),max(p2(ind2)),max(p2(ind3))];
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
   ind = find(f > 20 & f < 22);
   ind2 = find(f > 40 & f < 50);
     ind3 = find(f > 60 & f < 70);
    BaselineData = [BaselineData ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind)),max(p2(ind2)),max(p2(ind3))];
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
     ind = find(f > 20 & f < 22);
     
   ind2 = find(f > 40 & f < 50);
     ind3 = find(f > 60 & f < 70);
%      plot(f,p1)
        
    TestFeatureMatrix = [TestFeatureMatrix ; mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind)),max(p2(ind2)),max(p2(ind3))];
end


%% Fisher

GoodFeature=BaselineData;
BadFeature=DegradedData;
SelectNum=3;

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

TrainData=[BaselineData(:,Top);DegradedData(:,Top)];
TestData=TestFeatureMatrix(:,Top);
%% Train LR Model

%Label Vector (0.95 for good samples, 0.05 for bad samples
Label=[ones(size(BaselineData(:,Top),1),1)*0.95; ones(size(DegradedData(:,Top),1),1)*0.05];

%fit LR Model (glm-fit)
beta = glmfit([BaselineData(:,Top); DegradedData(:,Top)],Label,'binomial');

%% Calculating Health Value (using LR Model)


CV_Test = glmval(beta,TestFeatureMatrix(:,Top),'logit') ;  %Use LR Model
stem(CV_Test) 
 