%% Test 1 - Init
clear all; clc
load('cam1_1.mat')
load('cam2_1.mat')
load('cam3_1.mat')

% load the videos and transform them into 3D greyscale matrices
numFrames = size(vidFrames1_1,4);
for j = 1:numFrames
    X1(:,:,j) = im2double(rgb2gray(vidFrames1_1(:,:,:,j))); 
    X2(:,:,j) = im2double(rgb2gray(vidFrames2_1(:,:,:,j)));
    X3(:,:,j) = im2double(rgb2gray(vidFrames3_1(:,:,:,j))); 
end

%% Test 1 - Image processing
for i = 1:numFrames
    frame1 = X1(:,340-59:340+60,i);
    frame2 = X2(:,300-59:300+60,i);
    frame3 = X3(:,:,i) - mean(X3(:,:,i));
    frame3_filt = frame3(270-29:270+30,:);
    [m1, index1] = max(frame1(:));
    [m2, index2] = max(frame2(:));
    [m3, index3] = max(frame3_filt(:));
    s = [480 120];
    %imshow(frame3_filt)
    [y1(i),x1(i)] = ind2sub(s,index1);
    [y2(i),x2(i)] = ind2sub(s,index2);
    [y3(i),x3(i)] = ind2sub([60 640],index3);
end

%% Test 1 - plotting motion
lims = [0 500];
time_axis = linspace(1,numFrames,numFrames);
subplot(3,1,1);
plot(time_axis,y1,'b-','Linewidth',2)
hold on
plot(time_axis,x1,'r-','Linewidth',2)
ylim(lims); legend('y1','x1'); ylabel('Position'); xlabel('Time (s)'); title('Angle 1')
subplot(3,1,2);
plot(time_axis,y2,'b-','Linewidth',2)
hold on
plot(time_axis,x2,'r-','Linewidth',2)
ylim(lims); legend('y2','x2'); ylabel('Position'); xlabel('Time (s)'); title('Angle 2')
subplot(3,1,3);
plot(time_axis,y3,'b-','Linewidth',2)
hold on
plot(time_axis,x3,'r-','Linewidth',2)
ylim(lims); legend('y3','x3'); ylabel('Position'); xlabel('Time (s)'); title('Angle 3')

sgtitle('Vertical and Horizontal Motion vs Time')
print('-dpng','test1motion.png')



%% Test 1 - SVD and plotting
mean_vect = [mean(y1); mean(x1); mean(y2); mean(x2); mean(y3); mean(x3)];
X_matrix = [y1;x1;y2;x2;y3;x3] - mean_vect;
[U,S,V] = svd(X_matrix,'econ');
size(U)
size(S)
size(V) % captures the time information
v_trans = V';

% Compute energies
sig = diag(S);
for l = 1:6
    energy(l) = sig(l)^2/sum(sig.^2);
end
sum(energy(1:2))

% Plot energy graphs and dominant singular value position
subplot(2,1,1)
plot(linspace(1,6,6),energy,'o','Linewidth',2)
title('Energy vs Singular Values'); ylabel('Energy'); xlabel('Singular Values')
subplot(2,1,2)
plot(time_axis,v_trans(1,:),'b-','Linewidth',2)
hold on
plot(time_axis,v_trans(2,:),'r-','Linewidth',2)
ylabel('Displacement'); xlabel('Time (s)'); title('Mass Motion'); legend('Component 1','Component 2')


print('-dpng','test1svd.png')

%% Test 2 - init

% the shake introduces more noise in, so may need to consider more
% components

load('cam1_2.mat')
load('cam2_2.mat')
load('cam3_2.mat')
% note we cut off the last 40 or so frames from videos 2 and 3 (all same
% length of first video)
numFrames2 = size(vidFrames1_2,4);
for j = 1:numFrames2
    A1(:,:,j) = im2double(rgb2gray(vidFrames1_2(:,:,:,j))); 
    A2(:,:,j) = im2double(rgb2gray(vidFrames2_2(:,:,:,j)));
    A3(:,:,j) = im2double(rgb2gray(vidFrames3_2(:,:,:,j))); 
end
%% Test 2 - Image processing
for i = 1:numFrames2
    frame1a = A1(:,340-59:340+60,i);
    frame2a = A2(:,300-59:300+60,i);
    frame3a = A3(:,:,i) - mean(A3(:,:,i));
    frame3a_filt = frame3a(270-29:270+30,:);
    [m1, index1a] = max(frame1a(:));
    [m2, index2a] = max(frame2a(:));
    [m3, index3a] = max(frame3a_filt(:));
    s = [480 120];
    %imshow(frame3_filt)
    [y1a(i),x1a(i)] = ind2sub(s,index1a);
    [y2a(i),x2a(i)] = ind2sub(s,index2a);
    [y3a(i),x3a(i)] = ind2sub([60 640],index3a);
end

%% Test 2 - Motion plot
lims = [0 500];
time_axisa = linspace(1,numFrames2,numFrames2);
subplot(3,1,1);
plot(time_axisa,y1a,'b-','Linewidth',2)
hold on
plot(time_axisa,x1a,'r-','Linewidth',2)
ylim(lims); legend('y1','x1'); ylabel('Position'); xlabel('Time (s)'); title('Angle 1')
subplot(3,1,2);
plot(time_axisa,y2a,'b-','Linewidth',2)
hold on
plot(time_axisa,x2a,'r-','Linewidth',2)
ylim(lims); legend('y2','x2'); ylabel('Position'); xlabel('Time (s)'); title('Angle 2')
subplot(3,1,3);
plot(time_axisa,y3a,'b-','Linewidth',2)
hold on
plot(time_axisa,x3a,'r-','Linewidth',2)
ylim(lims); legend('y3','x3'); ylabel('Position'); xlabel('Time (s)'); title('Angle 3')

sgtitle('Vertical and Horizontal Motion vs Time')
print('-dpng','test2motion.png')

%% Test 2 - SVD and plotting
mean_vect = [mean(y1a); mean(x1a); mean(y2a); mean(x2a); mean(y3a); mean(x3a)];
A_matrix = [y1a;x1a;y2a;x2a;y3a;x3a] - mean_vect;
[Ua,Sa,Va] = svd(A_matrix,'econ');
va_trans = Va';

% Compute energies
siga = diag(Sa);
for l = 1:6
    energya(l) = siga(l)^2/sum(siga.^2);
end
sum(energya(1:3))

% Plot energy graphs and dominant singular value position
subplot(2,1,1)
plot(linspace(1,6,6),energya,'o','Linewidth',2)
title('Energy vs Singular Values'); ylabel('Energy'); xlabel('Singular Values')
subplot(2,1,2)
plot(time_axisa,va_trans(1,:),'Linewidth',3)
hold on
plot(time_axisa,va_trans(2,:),'Linewidth',2)
plot(time_axisa,va_trans(3,:),'Linewidth',1)
ylabel('Displacement'); xlabel('Time (s)'); title('Mass Motion'); legend('Component 1','Component 2','Component 3')

print('-dpng','test2svd.png')

%% Test 3 - Init
load('cam1_3.mat')
load('cam2_3.mat')
load('cam3_3.mat')
% note we cut off the last 40 or so frames from videos 2 and 3 (all same
% length of first video)
numFrames3 = size(vidFrames3_3,4);
for j = 1:numFrames3
    B1(:,:,j) = im2double(rgb2gray(vidFrames1_3(:,:,:,j))); 
    B2(:,:,j) = im2double(rgb2gray(vidFrames2_3(:,:,:,j)));
    B3(:,:,j) = im2double(rgb2gray(vidFrames3_3(:,:,:,j))); 
end

%% Test 3 - Image Processing
for i = 1:numFrames3
    frame1b = B1(:,340-59:340+60,i);
    frame2b = B2(:,300-59:300+60,i);
    frame3b = B3(:,:,i) - mean(B3(:,:,i));
    frame3b_filt = frame3b(270-69:270+70,:);
    [m1, index1b] = max(frame1b(:));
    [m2, index2b] = max(frame2b(:));
    [m3, index3b] = max(frame3b_filt(:));
    s = [480 120];
    %imshow(frame3_filt)
    [y1b(i),x1b(i)] = ind2sub(s,index1b);
    [y2b(i),x2b(i)] = ind2sub(s,index2b);
    [y3b(i),x3b(i)] = ind2sub([140 640],index3b);
end

%% Test 3 - Motion Plot
lims = [0 500];
time_axisb = linspace(1,numFrames3,numFrames3);
subplot(3,1,1);
plot(time_axisb,y1b,'b-','Linewidth',2)
hold on
plot(time_axisb,x1b,'r-','Linewidth',2)
ylim(lims); legend('y1','x1'); ylabel('Position'); xlabel('Time (s)'); title('Angle 1')
subplot(3,1,2);
plot(time_axisb,y2b,'b-','Linewidth',2)
hold on
plot(time_axisb,x2b,'r-','Linewidth',2)
ylim(lims); legend('y2','x2'); ylabel('Position'); xlabel('Time (s)'); title('Angle 2')
subplot(3,1,3);
plot(time_axisb,y3b,'b-','Linewidth',2)
hold on
plot(time_axisb,x3b,'r-','Linewidth',2)
ylim(lims); legend('y3','x3'); ylabel('Position'); xlabel('Time (s)'); title('Angle 3')

sgtitle('Vertical and Horizontal Motion vs Time')
print('-dpng','test3motion.png')

%% Test 3 - SVD and plotting
mean_vect = [mean(y1b); mean(x1b); mean(y2b); mean(x2b); mean(y3b); mean(x3b)];
B_matrix = [y1b;x1b;y2b;x2b;y3b;x3b] - mean_vect;
[Ub,Sb,Vb] = svd(B_matrix,'econ');
vb_trans = Vb';

% Compute energies
sigb = diag(Sb);
for l = 1:6
    energyb(l) = sigb(l)^2/sum(sigb.^2);
end
sum(energyb(1:4))

% Plot energy graphs and dominant singular value position
subplot(2,1,1)
plot(linspace(1,6,6),energyb,'o','Linewidth',2)
title('Energy vs Singular Values'); ylabel('Energy'); xlabel('Singular Values')
subplot(2,1,2)
plot(time_axisb,vb_trans(1,:),'Linewidth',4)
hold on
plot(time_axisb,vb_trans(2,:),'Linewidth',3)
plot(time_axisb,vb_trans(3,:),'Linewidth',2)
plot(time_axisb,vb_trans(4,:),'Linewidth',1)
ylabel('Displacement'); xlabel('Time (s)'); title('Mass Motion'); legend('Component 1','Component 2','Component 3','Component 4')

print('-dpng','test3svd.png')


%% Test 4 - Init
load('cam1_4.mat')
load('cam2_4.mat')
load('cam3_4.mat')
numFrames4 = size(vidFrames1_4,4);
for j = 1:numFrames4
    C1(:,:,j) = im2double(rgb2gray(vidFrames1_4(:,:,:,j))); 
    C2(:,:,j) = im2double(rgb2gray(vidFrames2_4(:,:,:,j)));
    C3(:,:,j) = im2double(rgb2gray(vidFrames3_4(:,:,:,j))); 
end

%% Test 4 - Image Processing
for i = 1:numFrames4
    frame1c = C1(:,340-59:340+60,i);
    frame2c = C2(:,300-59:300+60,i);
    frame3c = C3(:,:,i) - mean(C3(:,:,i));
    frame3c_filt = frame3c(270-79:270+80,:);
    [m1, index1c] = max(frame1c(:));
    [m2, index2c] = max(frame2c(:));
    [m3, index3c] = max(frame3c_filt(:));
    s = [480 120];
    %imshow(frame3_filt)
    [y1c(i),x1c(i)] = ind2sub(s,index1c);
    [y2c(i),x2c(i)] = ind2sub(s,index2c);
    [y3c(i),x3c(i)] = ind2sub([160 640],index3c);
end

%% Test 4 - Motion Plotting
lims = [0 500];
time_axisc = linspace(1,numFrames4,numFrames4);
subplot(3,1,1);
plot(time_axisc,y1c,'b-','Linewidth',2)
hold on
plot(time_axisc,x1c,'r-','Linewidth',2)
ylim(lims); legend('y1','x1'); ylabel('Position'); xlabel('Time (s)'); title('Angle 1')
subplot(3,1,2);
plot(time_axisc,y2c,'b-','Linewidth',2)
hold on
plot(time_axisc,x2c,'r-','Linewidth',2)
ylim(lims); legend('y2','x2'); ylabel('Position'); xlabel('Time (s)'); title('Angle 2')
subplot(3,1,3);
plot(time_axisc,y3c,'b-','Linewidth',2)
hold on
plot(time_axisc,x3c,'r-','Linewidth',2)
ylim(lims); legend('y3','x3'); ylabel('Position'); xlabel('Time (s)'); title('Angle 3')

sgtitle('Vertical and Horizontal Motion vs Time')
print('-dpng','test4motion.png')

%% Test 4 - SVD and plotting
mean_vect = [mean(y1c); mean(x1c); mean(y2c); mean(x2c); mean(y3c); mean(x3c)];
C_matrix = [y1c;x1c;y2c;x2c;y3c;x3c] - mean_vect;
[Uc,Sc,Vc] = svd(C_matrix,'econ');
vc_trans = Vc';

% Compute energies
sigc = diag(Sc);
for l = 1:6
    energyc(l) = sigc(l)^2/sum(sigc.^2);
end
sum(energyc(1:3))

% Plot energy graphs and dominant singular value position
subplot(2,1,1)
plot(linspace(1,6,6),energyc,'o','Linewidth',2)
title('Energy vs Singular Values'); ylabel('Energy'); xlabel('Singular Values')
subplot(2,1,2)
plot(time_axisc,vc_trans(1,:),'Linewidth',2)
hold on
plot(time_axisc,vc_trans(2,:),'Linewidth',1)
plot(time_axisc,vc_trans(3,:),'Linewidth',1)
ylabel('Displacement'); xlabel('Time (s)'); title('Mass Motion'); title('Mass Motion'); legend('Component 1','Component 2','Component 3')

print('-dpng','test4svd.png')

