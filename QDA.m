clc;clear all;
load /usr/local/MATLAB/S_Learning/vowel_train.txt;
load seamount
trainDataset = vowel_train(:,2:12);
clear vowel_train;
Traindim = size(trainDataset);
CLASS_NUMS = 11;
Mu = zeros (CLASS_NUMS,(Traindim(1,2)-1));
mainMu = zeros (1,(Traindim(1,2)-1));
%Sw = zeros ((Traindim(1,2)-1),(Traindim(1,2)-1));
Sb = zeros ((Traindim(1,2)-1),(Traindim(1,2)-1));

for k=1:CLASS_NUMS
    counter = 0;
    for i=1:Traindim(1,1)
        if (trainDataset(i,1) == k)
            counter = counter + 1;
            sortedDataset{k}(counter,:) = trainDataset(i,2:11);   
        end
    end
    Mu(k,:) = mean(sortedDataset{k}(:,:));
    Si{k} = cov(sortedDataset{k}(:,:));
    %Sw = Sw + Si{k}; 
end
for i=1:CLASS_NUMS
    ss = size(sortedDataset{i});
    mainMu = mainMu + (ss(1,1)) * Mu(i,:);
end

mainMu = mainMu / Traindim(1,1);

for i=1:CLASS_NUMS
    ss = size(sortedDataset{i});
    Sb = Sb + (ss(1,1)  * (Mu(i,:) - mainMu)' * (Mu(i,:) - mainMu));
end
for i=1:CLASS_NUMS
    J{i} = inv(Si{i})*Sb;
end

vars = var(trainDataset(:,2:11));

for i=1:CLASS_NUMS
    h = scatter(sortedDataset{i}(:,1),sortedDataset{i}(:,2),31);
    hold on;
end

min_y = zeros(1,CLASS_NUMS);
max_y = zeros(1,CLASS_NUMS);
y_Mu = zeros (CLASS_NUMS,1);
y_Si = zeros (CLASS_NUMS,1);

for i=1:CLASS_NUMS
    [V D] = eig(J{i});
    yy{i} = sortedDataset{i} * V(:,1);
    min_y(1,i) = min(yy{i});
    max_y(1,i) = max(yy{1});
    y_Mu(i,1) = mean(yy{i}(:,1));
    y_Si(i,1) = std(yy{i}(:,1));
end

F = mvnpdf(yy{1},(y_Mu(1,1)),(y_Si(1,1)));

figure
for i=1:(CLASS_NUMS-1)
    scatter(yy{i},mvnpdf(yy{i},(y_Mu(i,1)),(y_Si(i,1)))/2,CLASS_NUMS,'pentagram');
    hold on;
end

for i=1:CLASS_NUMS
    for j=1:CLASS_NUMS
        Prob_Calc{i}(:,j) = mvnpdf(yy{i},(y_Mu(j,1)),(y_Si(j,1)));
    end
end
aa = zeros(1,1);
ecounter = zeros(1,CLASS_NUMS);
for i=1:CLASS_NUMS
    for j=1:ss(1,1)
        aa = [aa;find(ismember(Prob_Calc{i}(j,:),max(Prob_Calc{i}(j,:))))];
        if ((find(ismember(Prob_Calc{i}(j,:),max(Prob_Calc{i}(j,:)))) ~= i) && (Prob_Calc{i}(j,i))<0.7)
            ecounter(1,i) = ecounter(1,i) + 1;            
        end
    end
end
TrainError = sum(ecounter) / Traindim(1,1)

              %%%%TEST DATASET%%%%%%%

load /usr/local/MATLAB/S_Learning/vowel_test.txt;
testDataset = vowel_test(:,2:12);
clear vowel_test;
Traindim = size(testDataset);
CLASS_NUMS = 11;
for k=1:CLASS_NUMS
    counter = 0;
    for i=1:Traindim(1,1)
        if (testDataset(i,1) == k)
            counter = counter + 1;
            TestsortedDataset{k}(counter,:) = testDataset(i,2:11);   
        end
    end
end
Testmin_y = zeros(1,CLASS_NUMS);
Testmax_y = zeros(1,CLASS_NUMS);
Testy_Mu = zeros (CLASS_NUMS,1);
Testy_Si = zeros (CLASS_NUMS,1);
for i=1:CLASS_NUMS
    Testyy{i} = TestsortedDataset{i} * V(:,1);
    Testmin_y(1,i) = min(Testyy{i});
    Testmax_y(1,i) = max(Testyy{1});
    Testy_Mu(i,1) = mean(Testyy{i}(:,1));
    Testy_Si(i,1) = std(Testyy{i}(:,1));
end
for i=1:CLASS_NUMS
    for j=1:CLASS_NUMS
        TestProb_Calc{i}(:,j) = mvnpdf(Testyy{i},(Testy_Mu(j,1)),(Testy_Si(j,1)));
    end
end
Testaa = zeros(1,1);
Testecounter = zeros(1,CLASS_NUMS);

for i=1:CLASS_NUMS
    ss = size(TestProb_Calc{i});
    for j=1:ss(1,1)
        Testaa = [Testaa;find(ismember(TestProb_Calc{i}(j,:),max(TestProb_Calc{i}(j,:))))];
        if ((find(ismember(TestProb_Calc{i}(j,:),max(TestProb_Calc{i}(j,:)))) ~= i) && (TestProb_Calc{i}(j,i))<0.7)
            Testecounter(1,i) = Testecounter(1,i) + 1;            
        end
    end
end
TestError = sum(Testecounter) / Traindim(1,1)
