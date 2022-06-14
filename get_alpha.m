function alpha=get_alpha(image)
% by Olga Dyakova, Yu-Jen Lee, Kit D. Longden, Valerij G. Kiselev & Karin Nordström
fft_image = fftshift(fft2(image));
abs_im = abs(fft_image);
[N,M] = size(abs_im);
[X,Y]=meshgrid(-M/2:M/2-1,-N/2:N/2-1);
[~,rho]=cart2pol(X,Y);
rho = round(rho);
abs_av=nan(1,floor(N/2));
for r = 0:floor(N/2-1)
    idx= rho==r;
    abs_av(r+1) = mean(abs_im(idx));
end

freq = 0:floor(N/2-1);
% figure
% loglog(freq,abs_av,'k')
% xlabel('spatial frequency')
% ylabel('average amplitude spectrum')
xx = log(freq(3:end));
yy = log(abs_av(freq(3:end)));

p = polyfit(xx,yy,1);
alpha = (-1)*p(1);
% display(alpha);