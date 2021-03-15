%% Init for videos
clear all; clc

monte = VideoReader('monte_carlo_low.mp4')
ski = VideoReader('ski_drop_low.mp4')
N = 540;
M = 960;
count = 1;
while hasFrame(monte)
    monteFrame = im2double(rgb2gray(readFrame(monte)));
    monte_dat(:,count) = reshape(monteFrame,[M*N 1]);
    count = count + 1;
end
count = 1;
while hasFrame(ski)
    skiFrame = im2double(rgb2gray(readFrame(ski)));
    ski_dat(:,count) = reshape(skiFrame,[M*N 1]);
    count = count + 1;
end
monte_length = size(monte_dat,2);
ski_length = size(ski_dat,2);
% each column of monte_dat and ski_dat is a time evolution

%% Set up X matrices and take SVD
% chan change monte_dat for ski_dat to switch between videos
X1 = ski_dat(:,1:end-1);
X2 = ski_dat(:,2:end);
[U, Sigma, V] = svd(X1,'econ');

%% Plot singular values and determine necessary modes
sing_vals = diag(Sigma);
plot(linspace(1,length(sing_vals),length(sing_vals)),sing_vals,'o')
title('Singular Values for Monte Carlo')
print('-dpng','svd.png')

% only use the significant modes
modes = 2;
U_trunc = U(:,1:modes);
Sigma_trunc = Sigma(1:modes,1:modes);
V_trunc = V(:,1:modes);

%% DMD Init
dt = 1/ski.FrameRate;

S_tilde = U_trunc'*X2*V_trunc/Sigma_trunc;

[eV, D] = eig(S_tilde);
Phi = X2*V_trunc/Sigma_trunc*eV;
mu = diag(D);
omega = log(mu)/dt;

%% DMD Reconstruction
b = Phi\X1(:,1);
iter_length = size(X1,2);
time_vect = 0:dt:iter_length-1;

dynamics = zeros(modes,iter_length);
for iter = 1:iter_length
    dynamics(:,iter) = b.*exp(omega*time_vect(iter));
end
X_dmd = Phi*dynamics;

%% Sparse and nonsparse construction
X_sparse = X1-abs(X_dmd)+0.5; % computes real-valued sparse matrix (add 0.5 to this for ski drop contrast)
R = X_sparse.*(X_sparse<0); % create residual matrix of negative values

background = R + abs(X_dmd);
foreground = X_sparse - R;

X_approx = background + foreground;

%% Reshape into videos
dur = ski_length - 1; % will need to change to ski_length for other video
bg_vid = reshape(background, [N, M, dur]);
fg_vid = reshape(foreground, [N, M, dur]);
dmd_vid = reshape(X_approx, [N, M, dur]);

%% Play videos
for i = 1:dur
    imshow(fg_vid(:,:,i))
end

%% Capture images
subplot(1,2,1)
imshow(reshape(ski_dat(:,50),[N M]))
title('Original Video','Fontsize',14)
subplot(1,2,2)
imshow(bg_vid(:,:,50))
title('Background Video','Fontsize',14)
print('-dpng','ski_hor.png')

%%
imshow(fg_vid(:,:,50))
title('Foreground Video','Fontsize',18)
print('-dpng','iso_hor.png')
