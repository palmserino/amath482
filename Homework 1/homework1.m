%% Part 0 - Initialize
% Clean workspace
clear all; close all; clc

load subdata.mat % Imports the data as the 262144x49 (space by time) matrix called subdata

L = 10; % spatial domain
n = 64; % Fourier modes

x2 = linspace(-L,L,n+1); x = x2(1:n); y =x; z = x;
k = (2*pi/(2*L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);

[X,Y,Z]=meshgrid(x,y,z);
[Kx,Ky,Kz]=meshgrid(ks,ks,ks);


%% Part 1 - average 
avg = zeros(n,n,n);
for j=1:49
    Un(:,:,:)=reshape(subdata(:,j),n,n,n);
    unf = fftn(Un);
    avg = avg + unf;
end
avg = abs(fftshift(avg))/49;
oneD = avg(:);
% Find the center frequency
[val,ind] = max(oneD)
s = [64,64,64];
[i,j,k] = ind2sub(s,ind) % prints indices of the max
% Central frequencies
kx_center = Kx(i,j,k)
ky_center = Ky(i,j,k)
kz_center = Kz(i,j,k)

%% Part 2 - filter around the central freq
a = 2; % window size
filter = exp(-a*((Kx-kx_center).^2 + (Ky-ky_center).^2 + (Kz-kz_center).^2));

for index = 1:49
    Un(:,:,:)=reshape(subdata(:,index),n,n,n);
    unf = fftn(Un);
    un_filt = fftshift(unf).*filter; 
    un = abs(ifftn(fftshift(un_filt))); % 64 x 64 x 64 (remove n for good curve)
    un_oneD = un(:);
    [val,ind] = max(un_oneD);
    [i(index),j(index),k(index)] = ind2sub(s,ind); % max indices for 1 time?
end
in = [i', j', k']; % gives a 49 x 3 matrix (49 times values in rows and indices in columns)

final_coords = [x(in(end,1)), y(in(end,2)), z(in(end,3))] % prints the final x, y, z coords

% Plot the path of submarine
plot3(x(in(:,1)),y(in(:,2)),z(in(:,3)),'k-o','LineWidth',2)
xlabel('x')
ylabel('y')
zlabel('z')
axis([-L L -L L -L L])
title('Submarine Path')
print('sub_path.png','-dpng')


%% Part 3 - generate x and y coordinate table
pos(:,1) = x(in(:,1));
pos(:,2) = y(in(:,2));
pos = table(pos);
writetable(pos,'positions.csv')



