
%% Part 1a - GNR (only Guitar), once have spectrogram try log and the bass/guitar line will be very clear
clear all; clc
[y, Fs] = audioread('GNR.m4a'); % note that Fs is the sample rate (Hz)
trgnr = length(y)/Fs; % record time in seconds

L = trgnr;
n = length(y);
k = (2*pi/(L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
t = (1:length(y))/Fs;
tau = 0:0.05:L;


figure(1)
a = 100;
tic
for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2);
    Sg = g.*y';
    Sgt = fft(Sg);
    Sgtspec(:,i) = fftshift(abs(Sgt));
end
toc
pcolor(tau,ks/(2*pi),Sgtspec)
shading interp
set(gca,'Ylim',[0 500],'Fontsize',12)
colormap(hot)
colorbar
title('GNR Spectrogram')
xlabel('time (s)'), ylabel('frequency (Hz)')
print('gnr_spec_new.png','-dpng')

%% Part 1b - Floyd bass
clear all; clc
figure(2)
[y, Fs] = audioread('Floyd.m4a'); % note that Fs is the sample rate (Hz)
y = y(1:((length(y)-1)/5)); % look at the first tenth of the clip (due to memory issues)
trgnr = (length(y))/Fs; % record time in seconds
L = trgnr;
n = length(y);
k = (2*pi/(L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
t = (1:n)/Fs;
tau = 0:0.1:L;
size(t)
size(y')

a = 100;
tic
for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2);
    Sg = g.*y';
    Sgt = fft(Sg);
    Sgtspec(:,i) = fftshift(abs(Sgt));
end
toc
pcolor(tau,ks/(2*pi),Sgtspec)
shading interp
set(gca,'Ylim',[0 500],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (s)'), ylabel('frequency (Hz)')
title('Floyd Spectrogram')
print('floyd_spec_new.png','-dpng')


%% Part 2 - Applying low pass filter to Floyd (isolate bass)
clear all; clc
figure(3)
[y, Fs] = audioread('Floyd.m4a'); % note that Fs is the sample rate (Hz)
y = y(1:((length(y)-1)/5)); % look at the first tenth of the clip (due to memory issues)
trgnr = (length(y))/Fs; % record time in seconds
L = trgnr;
n = length(y);
k = (2*pi/(L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
t = (1:n)/Fs;
tau = 0:0.1:L;
size(t)
size(y')

a = 100;
tic
for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2);
    Sg = g.*y';
    Sgt = fft(Sg);
    for j = 1:length(Sgt)
        freq = k(j)/(2*pi);
        if abs(freq) > 200
            Sgt(j) = 0;
        end
    end
    Sgtspec(:,i) = fftshift(abs(Sgt));
end
toc
pcolor(tau,ks/(2*pi),Sgtspec)
shading interp
set(gca,'Ylim',[0 200],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (s)'), ylabel('frequency (Hz)')
title('Floyd Spectrogram filtered around bass')
print('-dpng','floyd_bass_filt.png')



%% Part 3 - Recreate the guitar solo in Comfortably Numb (high pass filter)
clear all; clc
figure(4)
[y, Fs] = audioread('Floyd.m4a'); % note that Fs is the sample rate (Hz)
y = y(1:((length(y)-1)/5)); % look at the first tenth of the clip (due to memory issues)
trgnr = (length(y))/Fs; % record time in seconds
L = trgnr;
n = length(y);
k = (2*pi/(L))*[0:(n/2 - 1) -n/2:-1]; ks = fftshift(k);
t = (1:n)/Fs;
tau = 0:0.1:L;
size(t)
size(y')

a = 100;
tic
for i = 1:length(tau)
    g = exp(-a*(t-tau(i)).^2);
    Sg = g.*y';
    Sgt = fft(Sg);
    for j = 1:length(Sgt)
        freq = k(j)/(2*pi);
        if abs(freq) < 300 || abs(freq) > 600
            Sgt(j) = 0;
        end
    end
    Sgtspec(:,i) = fftshift(abs(Sgt));
end
toc
pcolor(tau,ks/(2*pi),Sgtspec)
shading interp
set(gca,'Ylim',[300 600],'Fontsize',12)
colormap(hot)
colorbar
xlabel('time (s)'), ylabel('frequency (Hz)')
title('Floyd Guitar Spectrogram')
print('-dpng','floyd_guitar_filt.png')

