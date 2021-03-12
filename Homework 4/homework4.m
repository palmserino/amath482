
%% Part 0 - Init and process
clear all; clc
% can load test data by changing train  to test
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
num_img = size(images,3);
data = zeros(28*28,num_img);
for i = 1:num_img
    data(:,i) = reshape(images(:,:,i),[28*28 1]);
end

%% Part 1 - Perform SVD
[U,S,V] = svd(data,'econ');

%% Part 1 - Compute energy and 90% r value (choose 64 for simplicity)
sig = diag(S);
for l = 1:length(sig)
    energy(l) = sig(l)^2/sum(sig.^2);
end

r = 64;
sum(energy(1:r))

%% Part 1 - plot singular value spectrum
sig_vect = linspace(0,length(sig),length(sig));
subplot(2,1,1)
semilogy(sig_vect,sig,'ro')
xlabel('singular values'); ylabel('value'); title('Log plot of singular values')
subplot(2,1,2)
plot(sig_vect,energy,'bo')
xlabel('singular values'); ylabel('energy'); title('Energy of singular values')
sgtitle('Singular Value Spectrum')
print('-dpng','sing_spect.png')

%% Part 1 - project onto 3 V-modes
% create indices vectors for each digit
zero = find(labels==0);
one = find(labels==1);
two = find(labels==2);
three = find(labels==3);
four = find(labels==4);
five = find(labels==5);
six = find(labels==6);
seven = find(labels==7);
eight = find(labels==8);
nine = find(labels==9);

% modes
m1 = 2;
m2 = 3;
m3 = 5;

% Projections and plot
cmap = colormap(parula(10));

project0a = V(zero,m1)'*data(:,zero)'; project0b = V(zero,m2)'*data(:,zero)'; project0c = V(zero,m3)'*data(:,zero)';
plot3(project0a,project0b,project0c,'.','Color',cmap(1,:))
hold on

project1a = V(one,m1)'*data(:,one)'; project1b = V(one,m2)'*data(:,one)'; project1c = V(one,m3)'*data(:,one)';
plot3(project1a,project1b,project1c,'.','Color',cmap(2,:))

project2a = V(two,m1)'*data(:,two)'; project2b = V(two,m2)'*data(:,two)'; project2c = V(two,m3)'*data(:,two)';
plot3(project2a,project2b,project2c,'.','Color',cmap(3,:))

project3a = V(three,m1)'*data(:,three)'; project3b = V(three,m2)'*data(:,three)'; project3c = V(three,m3)'*data(:,three)';
plot3(project3a,project3b,project3c,'.','Color',cmap(4,:))

project4a = V(four,m1)'*data(:,four)'; project4b = V(four,m2)'*data(:,four)'; project4c = V(four,m3)'*data(:,four)';
plot3(project4a,project4b,project4c,'.','Color',cmap(5,:))

project5a = V(five,m1)'*data(:,five)'; project5b = V(five,m2)'*data(:,five)'; project5c = V(five,m3)'*data(:,five)';
plot3(project5a,project5b,project5c,'.','Color',cmap(6,:))

project6a = V(six,m1)'*data(:,six)'; project6b = V(six,m2)'*data(:,six)'; project6c = V(six,m3)'*data(:,six)';
plot3(project6a,project6b,project6c,'.','Color',cmap(7,:))

project7a = V(seven,m1)'*data(:,seven)'; project7b = V(seven,m2)'*data(:,seven)'; project7c = V(seven,m3)'*data(:,seven)';
plot3(project7a,project7b,project7c,'.','Color',cmap(8,:))

project8a = V(eight,m1)'*data(:,eight)'; project8b = V(eight,m2)'*data(:,eight)'; project8c = V(eight,m3)'*data(:,eight)';
plot3(project8a,project8b,project8c,'.','Color',cmap(9,:))

project9a = V(nine,m1)'*data(:,nine)'; project9b = V(nine,m2)'*data(:,nine)'; project9c = V(nine,m3)'*data(:,nine)';
plot3(project9a,project9b,project9c,'.','Color',cmap(10,:))

legend('0','1','2','3','4','5','6','7','8','9')
title(['Digit Projections onto V-modes ',num2str(m1),', ',num2str(m2),', and ',num2str(m3)])

print('-dpng','mode_projection.png')

%% Part 2 - LDA

[images, labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
num_img = size(images,3);
data = zeros(28*28,num_img);
for i = 1:num_img
    data(:,i) = reshape(images(:,:,i),[28*28 1]);
end
[U,S,V] = svd(data,'econ'); 
zero = find(labels==0);
one = find(labels==1);
two = find(labels==2);
three = find(labels==3);
four = find(labels==4);
five = find(labels==5);
six = find(labels==6);
seven = find(labels==7);
eight = find(labels==8);
nine = find(labels==9);
r = 64;
% LDA between 0 and 1 (with r modes from part 1)
new_data = S*V';
zeroS = new_data(1:r,zero);
oneS = new_data(1:r,one);

n0 = size(zeroS,2);
n1 = size(oneS,2);

md = mean(zeroS,2);
mc = mean(oneS,2);

Sw = 0; % within class variances
for k = 1:n0
    Sw = Sw + (zeroS(:,k) - md)*(zeroS(:,k) - md)';
end
for k = 1:n1
    Sw =  Sw + (oneS(:,k) - mc)*(oneS(:,k) - mc)';    
end

Sb = (md-mc)*(md-mc)'; % between class

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);
vzeros = w'*zeroS;
vones = w'*oneS;
sort0 = sort(vzeros);
sort1 = sort(vones);
t1 = length(sort0);
t2 = 1;
while sort0(t1) > sort1(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort0(t1) + sort1(t2))/2;
count_zero = 0;
for i = 1:length(vzeros)
    if (vzeros(i) < threshold)
        count_zero = count_zero + 1;
    end
end
count_zero/length(vzeros)
count_ones = 0;
for i = 1:length(vones)
    if (vones(i) > threshold)
        count_ones = count_ones + 1;
    end
end
count_ones/length(vones)
        

%% Part 2 - Plot the 0 and 1 LDA results
plot(vzeros,zeros(n0),'ob','Linewidth',2)
hold on
plot(vones,ones(n1),'or','Linewidth',2)
ylim([0 1.2]); title('LDA on Zeros and Ones'); xlabel('Left = more 0, Right = more 1')
yticks([0 1]); yticklabels({'Zeros','Ones'})

print('-dpng','zero_one_LDA.png')

%% Part 3 - Make histogram for easiest to seperate part

subplot(1,2,1)
histogram(sort(vzeros)); hold on, plot([threshold threshold], [0 160],'r','Linewidth',2)
title('Zeros')
subplot(1,2,2)
histogram(sort(vones)); hold on, plot([threshold threshold], [0 180],'r','Linewidth',2)
title('Ones')

print('-dpng','hist_0_1.png')

%% Part 2 - LDA with three variables
% LDA between 0, 1, and 5 (with r modes from part 1)
new_data = S*V';
zeroS = new_data(1:r,zero);
oneS = new_data(1:r,one);
fiveS = new_data(1:r,five);

n0 = size(zeroS,2);
n1 = size(oneS,2);
n5 = size(fiveS,2);

m0 = mean(zeroS,2);
m1 = mean(oneS,2);
m5 = mean(fiveS,2);
m = (m0+m1+m5)/3;

Sw = 0; % within class variances
for k = 1:n0
    Sw = Sw + (zeroS(:,k) - m0)*(zeroS(:,k) - m0)';
end
for k = 1:n1
    Sw =  Sw + (oneS(:,k) - m1)*(oneS(:,k) - m1)';    
end
for k = 1:n5
    Sw =  Sw + (fiveS(:,k) - m5)*(fiveS(:,k) - m5)';    
end
sb1 = (m0-m)*(m0-m)';
sb2 = (m1-m)*(m1-m)';
sb3 = (m5-m)*(m5-m)';
Sb = (sb1+sb2+sb3)/3;

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);
vzeros = w'*zeroS;
vones = w'*oneS;
vfives = w'*fiveS;

vzeros = sort(vzeros); vones = sort(vones); vfives = sort(vfives);

%% Plot the 3 LDA
plot(vzeros,zeros(n0),'ob','Linewidth',2)
hold on
plot(vones,ones(n1),'or','Linewidth',2)
plot(vfives,2*ones(n5),'og','Linewidth',2)
ylim([0 2.4]); title('LDA on Zeros, Ones, and Fives');
yticks([0 1 2]); yticklabels({'Zeros','Ones','Fives'})
print('-dpng','lda3cases.png')

%% Part 3 - Hardest digits to seperate
% LDA between 4 and 9, I use all the zero and one nomenclature to speed up
% the code writing process! (zeroS would be analagous to fourS)
new_data = S*V';
zeroS = new_data(1:r,four);
oneS = new_data(1:r,nine);

n0 = size(zeroS,2);
n1 = size(oneS,2);

md = mean(zeroS,2);
mc = mean(oneS,2);

Sw = 0; % within class variances
for k = 1:n0
    Sw = Sw + (zeroS(:,k) - md)*(zeroS(:,k) - md)';
end
for k = 1:n1
    Sw =  Sw + (oneS(:,k) - mc)*(oneS(:,k) - mc)';    
end

Sb = (md-mc)*(md-mc)'; % between class

[V2, D] = eig(Sb,Sw); % linear disciminant analysis
[lambda, ind] = max(abs(diag(D)));
w = V2(:,ind);
w = w/norm(w,2);
vzeros = w'*zeroS;
vones = w'*oneS;
sort0 = sort(vzeros);
sort1 = sort(vones);
t1 = length(sort0);
t2 = 1;
while sort0(t1) > sort1(t2)
    t1 = t1 - 1;
    t2 = t2 + 1;
end
threshold = (sort0(t1) + sort1(t2))/2;
count_zero = 0;
for i = 1:length(vzeros)
    if (vzeros(i) < threshold)
        count_zero = count_zero + 1;
    end
end
count_zero/length(vzeros)
count_ones = 0;
for i = 1:length(vones)
    if (vones(i) > threshold)
        count_ones = count_ones + 1;
    end
end
count_ones/length(vones)

%% Part 3 - Make histogram for hardest to seperate part
figure1 = figure('Position', [100, 100, 1600, 600]);
subplot(1,3,1)
plot(vzeros,zeros(n0),'ob','Linewidth',2)
hold on
plot(vones,ones(n1),'or','Linewidth',2)
ylim([0 1.2]); title('LDA on Fours and Nines'); xlabel('Left = more 4, Right = more 9')
yticks([0 1]); yticklabels({'Fours','Nines'})
threshold = (sort0(t1) + sort1(t2))/2;
subplot(1,3,2)
histogram(sort(vzeros)); hold on, plot([threshold threshold], [0 180],'r','Linewidth',2)
title('Fours')
subplot(1,3,3)
histogram(sort(vones)); hold on, plot([threshold threshold], [0 160],'r','Linewidth',2)
title('Nines')

print('-dpng','plots_4_9.png')

%% Part 4 - Load in training data
% Load
[images, labels] = mnist_parse('train-images.idx3-ubyte', 'train-labels.idx1-ubyte');
num_img = size(images,3);
data = zeros(28*28,num_img);
for i = 1:num_img
    data(:,i) = reshape(images(:,:,i),[28*28 1]);
end

zero = find(labels==0);
one = find(labels==1);
four = find(labels==4);
nine = find(labels==9);

% Compute SVD and project
r = 64;
[U,S,V] = svd(data,'econ'); 
sv = S*V';
proj_data = sv ./ max(sv(:));
proj_data = proj_data(1:r,:);

svm_dat = cat(2,proj_data(:,four),proj_data(:,nine));
svm_lab = cat(1,labels(four),labels(nine));

%% Compute Decision tree and SVM on training data
tree=fitctree(proj_data',labels);
%%
Mdl = fitcecoc(svm_dat',svm_lab);

%% Predict with test data
% Load in test data
[images, labels] = mnist_parse('t10k-images.idx3-ubyte', 't10k-labels.idx1-ubyte');
num_img = size(images,3);
data = zeros(28*28,num_img);
for i = 1:num_img
    data(:,i) = reshape(images(:,:,i),[28*28 1]);
end

% Compute SVD and project
[U,S,V] = svd(data,'econ'); 
sv = S*V';
proj_data = sv ./ max(sv(:));
proj_data = proj_data(1:r,:);

% Redefine indices for test data
zero = find(labels==0);
one = find(labels==1);
four = find(labels==4);
nine = find(labels==9);

svm_lab = cat(1,labels(zero),labels(one));
svm_dat = cat(2,proj_data(:,four),proj_data(:,nine));

%% Test decision tree and compute accuracies
test_tree = predict(tree,proj_data');
count_zero = 0;
tester = test_tree(four);
for i = 1:length(tester)
    if (tester(i) == 4)
        count_zero = count_zero + 1;
    end
end
count_zero / length(tester)

count_ones = 0;
tester = test_tree(nine);
for i = 1:length(tester)
    if (tester(i) == 9)
        count_ones = count_ones + 1;
    end
end
count_ones / length(tester)

%% Test SVM data
test_svm = predict(Mdl,svm_dat');

zeroSV = 1:length(proj_data(zero));
oneSV = length(zero)+1:length(svm_dat);
fourSV = 1:length(proj_data(four));
nineSV = length(four)+1:length(svm_dat);

count_zero = 0;
tester = test_svm(fourSV);
for i = 1:length(tester)
    if (tester(i) == 4)
        count_zero = count_zero + 1;
    end
end
count_zero / length(tester)

count_ones = 0;
tester = test_svm(nineSV);
for i = 1:length(tester)
    if (tester(i) == 9)
        count_ones = count_ones + 1;
    end
end
count_ones / length(tester)

