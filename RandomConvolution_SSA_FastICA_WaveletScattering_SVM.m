clear all;clc;
signal1 = xlsread('C:\Fan_1_normal_humid_room_data.xlsx');  
signal2 = xlsread('C:\Fan_2_arc_humid_45_data.xlsx');  
label=cell(89,1);
for i =1:50
    label{i,1}='class 1';
end
for i =51:89
    label{i,1}='class 2';
end
Data=[signal1';signal2'];  
for i =1:size(Data,1)
    Data(i,:)=timeseriesnormalize(Data(i,:));
end
percent_train = 70;
numberOfData=89;
randIndex=randperm(numberOfData);
randpermData=Data(randIndex,:);
randpermLabel=label(randIndex);
tranNumber=floor(numberOfData*percent_train/100);  
trainData=randpermData(1:tranNumber,:);%ECGData.Data(1:tranNumber,:)
testData=randpermData(tranNumber+1:numberOfData,:);
trainLabels=randpermLabel(1:tranNumber,:);%ECGData.Labels(1:tranNumber,:)
testLabels=randpermLabel(tranNumber+1:numberOfData,:);
Ctest = countcats(categorical(testLabels))./numel(testLabels).*100;
Ctrain = countcats(categorical(trainLabels))./numel(trainLabels).*100;
%helperPlotRandomRecords(ECGData,14)
N = 90002;
sn = waveletScattering(SignalLength=N,InvarianceScale=150, SamplingFrequency=128);
[fb,f,filterparams] = filterbank(sn);
figure
tiledlayout(2,1)
nexttile
plot(f,fb{2}.psift)
xlim([0 128])
grid on
title("1st Filter Bank Wavelet Filters")
nexttile
plot(f,fb{3}.psift)
xlim([0 128])
grid on
title("2nd Filter Bank Wavelet Filters")
xlabel("Hz")

scat_features_train = featureMatrix(sn,trainData');
Nwin = size(scat_features_train,2);
scat_features_train = permute(scat_features_train,[2 3 1]);
scat_features_train = reshape(scat_features_train, size(scat_features_train,1)*size(scat_features_train,2),[]);

scat_features_test = featureMatrix(sn,testData');
scat_features_test = permute(scat_features_test,[2 3 1]);
scat_features_test = reshape(scat_features_test, size(scat_features_test,1)*size(scat_features_test,2),[]);

sequence_labels_train =createSequenceLabels(Nwin,trainLabels);  
sequence_labels_test = createSequenceLabels(Nwin,testLabels);
scat_features = [scat_features_train; scat_features_test];
allLabels_scat = [sequence_labels_train; sequence_labels_test];
rng(1);
template = templateSVM(...
    KernelFunction="polynomial", ...
    PolynomialOrder=2, ...
    KernelScale="auto", ...
    BoxConstraint=1, ...
    Standardize=true);
classificationSVM = fitcecoc(...
    scat_features, ...
    allLabels_scat, ...
    Learners=template, ...
    Coding="onevsone", ...
    ClassNames={'class 1';'class 2'});
kfoldmodel = crossval(classificationSVM,KFold=5);

predLabels = kfoldPredict(kfoldmodel);
loss = kfoldLoss(kfoldmodel)*100;
confmatCV = confusionmat(allLabels_scat,predLabels)

fprintf("Accuracy is %2.2f percent.\n",100-loss);

classes = categorical({'class 1','class 2'});
[ClassVotes,ClassCounts] = helperMajorityVote(predLabels,[trainLabels; testLabels],classes);

CVaccuracy = sum(eq(ClassVotes,categorical([trainLabels; testLabels])))/162*100;
fprintf("True cross-validation accuracy is %2.2f percent.\n",CVaccuracy);

MVconfmatCV = confusionmat(categorical([trainLabels; testLabels]),ClassVotes);

model = fitcecoc(...
     scat_features_train, ...
     sequence_labels_train, ...
     Learners=template, ...
     Coding="onevsone", ...
     ClassNames={'class 1','class 2'});
predLabels = predict(model,scat_features_test);
[TestVotes,TestCounts] = helperMajorityVote(predLabels,testLabels,classes);
testaccuracy = sum(eq(TestVotes,categorical(testLabels)))/numel(testLabels)*100;
fprintf("The test accuracy is %2.2f percent. \n",testaccuracy);
model = fitcecoc(...
     scat_features_train, ...
     sequence_labels_train, ...
     Learners=template, ...
     Coding="onevsone", ...
     ClassNames={'class 1','class 2'});
predLabels = predict(model,scat_features_test);
[TestVotes,TestCounts] = helperMajorityVote(predLabels,testLabels,classes);
testaccuracy = sum(eq(TestVotes,categorical(testLabels)))/numel(testLabels)*100;
fprintf("The test accuracy is %2.2f percent. \n",testaccuracy);



signal1=signal1';
signal2=signal2';
mixSignalTemp=[];
mixSignal=[]
L = 180;
R=4;
for i=1:10
    for j=1:10
        mixSignalTemp=0.9684*signal1(i,:)+0.0316*signal2(j,:);    %alpha1=0.9684,alpha2=0.0316
        if i==1 & j==1
            mixSignalforPlot=mixSignalTemp;
        end
        mixSignal=[mixSignal;mixSignalTemp];
    end
end
for i=1:100
    mixSignal(i,:)=timeseriesnormalize(mixSignal(i,:));
end

scat_features_mixed = featureMatrix(sn,mixSignal');
Nwin = size(scat_features_mixed,2);
scat_features_mixed = permute(scat_features_mixed,[2 3 1]);
scat_features_mixed = reshape(scat_features_mixed, size(scat_features_mixed,1)*size(scat_features_mixed,2),[]);
predLabels_mixed= predict(model,scat_features_mixed);

count=0;
for i=1:100
    if predLabels_mixed(i)=="class 1"
        count=count+1;
    end
end
fprintf("correct classifiered sample numuber by directed ScatteringTransform_SVM: %4.0f \n",100-count);

RandomICA_separatedSignal=[];
for i =1:100 
    mixSignalForRandomICA=[];
    for j=1:2 
        convOperater1= rand(1, 100);
        tempsignal=conv(mixSignal(i,:),convOperater1);
        tempsignal=SSA(tempsignal,L,R)';
        mixSignalForRandomICA=[mixSignalForRandomICA;tempsignal];
    end
    RandomICA_separatedSignal=[RandomICA_separatedSignal;mixSignalForRandomICA];
end
RandomICA_separatedSignal=RandomICA_separatedSignal(:,1:90002);
RandomICAcount=0;

%filteredSignal=SSA(Signal,L,R);
for i=1:100
    fprintf("Implementing the fastICA: %4.0f \n",i);
    Z = FastICA(RandomICA_separatedSignal(2*(i-1)+1:2*i,:));   
      
    scat_features_demixed = featureMatrix(sn,Z');
    Nwin = size(scat_features_demixed,2);
    scat_features_demixed = permute(scat_features_demixed,[2 3 1]);
    scat_features_demixed = reshape(scat_features_demixed, size(scat_features_demixed,1)*size(scat_features_demixed,2),[]);
    predLabels_demixed= predict(model,scat_features_demixed);
    tempCount=0;
    for j=1:2  %8
        if predLabels_demixed(j)=="class 2"
            tempCount=tempCount+1;
        end
    end
    if tempCount>=1
        RandomICAcount=RandomICAcount+1;
    end
end
fprintf("correct classifiered sample numuber: %4.0f \n",RandomICAcount);


