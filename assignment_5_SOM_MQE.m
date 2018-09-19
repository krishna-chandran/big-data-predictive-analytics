

clear all;close all;clc


%% Getting degraded training data

cd 'D:\Alive\IMS\Big data\homework 3\Training\Faulty';
d=dir('*.txt');
 
numfiles = length(d);

DegradedData = [];
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
        
    DegradedData(k,:)= [mean_val, variance,skew_val,kurtosis_val,rms_val,peak_val,max(p2(ind1)),max(p2(ind2)),max(p2(ind3))];
 end
 
 

 
%% Getting healthy training data

cd 'D:\Alive\IMS\Big data\homework 3\Training\Healthy';
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
TrainData=[BaselineData(:,Top);DegradedData(:,Top)];
TestData=TestFeatureMatrix(:,Top);


%%


 allfeatures=TrainData;

 sMapt = som_make(allfeatures)
 som_show(sMapt,'umat','all')

   test1label={'Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy'}
    test1labe2={'Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2'}
   testlabel=vertcat(test1label', test1labe2');


sData = som_data_struct(TrainData, 'name', 'BearingData', ...
    'labels', testlabel);

sMapt = som_make(sData)
% som_show(sMapt,'umat','all')
    
sTo = som_autolabel(sMapt, sData)
%  som_show(sTo,'umat','all','empty','Labels')
som_show(sTo,'umat','all','empty','Labels')

% figure(1)
 som_show_add('label',sTo,'Textsize',10,'TextColor','r','Subplot',2)
% tilte('SOM Map - Train data')

%%
allfeatures=TestData;
 sMapt = som_make(allfeatures)
 som_show(sMapt,'umat','all')

   test1label={'Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy','Healthy'}
    test1labe2={'Unbalance Level  1','Unbalance Level  1','Unbalance Level  1','Unbalance Level  1','Unbalance Level  1','Unbalance Level  1','Unbalance Level  1','Unbalance Level  1','Unbalance Level  1','Unbalance Level  1'}
    test1labe3={'Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2','Unbalance Level  2'}
 
    
    testlabel=[test1label';test1labe2';test1labe3'];


sData = som_data_struct(allfeatures, 'name', 'BearingData', ...
    'labels', testlabel);

sMapt = som_make(sData)
% som_show(sMapt,'umat','all')
    
sTo = som_autolabel(sMapt, sData)
%  som_show(sTo,'umat','all','empty','Labels')
som_show(sTo,'umat','all','empty','Labels')

figure(1)
som_show_add('label',sTo,'Textsize',10,'TextColor','r','Subplot',2)
% tilte('SOM Map - Test data')


% som_show(sMap);
%% Calculate the MQE values for the testing data set

allfeatures=TrainData(1:10,:);

 sMapt = som_make(allfeatures)

S=size(TestData);
S=S(1);
for ii=1:S
    [qe te]=som_quality(sMapt,TestData(ii,:)); % calculate MQE value for each sample
    MQEt(ii)=qe;
end

MQEtn=(1-(MQEt)./(max(MQEt))); % normalize MQE
MQEtn=MQEtn';

%% Plot the calculated MQE values 
% observe the difference between normal condition and faluty conditions
figure(2)
plot(MQEtn,'-*r')
xlabel('Data file No.');
 ylabel('Confidence value (MQE)');
 title('Health Assessment Plot- MQE');


