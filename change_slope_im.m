% function f=change_slope_im(image, alpha, new_file_name)
function f=change_slope_im(image, alpha)

L = min(size(image));
image = imresize(image, [L L], 'bicubic');
fft_image = fftshift(fft2(image));
abs_im = abs(fft_image);
freq = -L/2:L/2-1;
[x, y] = meshgrid(freq, freq);
[~, ro] = cart2pol(x,y);
ro=round(ro);

abs_im_av = zeros(L,L);
for r = 0:L-1
    idx= ro==r;
    temp = mean(abs_im(idx));
    abs_im_av(idx) = temp;
end

zeroslope = fft_image./((abs_im_av));
filter_image=zeroslope.*(1+ro).^(-alpha);
filter_image_shiftback=ifftshift(filter_image);
image_new = real(ifft2(filter_image_shiftback));
% f = uint8(((image_new-min(image_new(:)))./((max(image_new(:)))-min(image_new(:)))).*255);
f = ((image_new-min(image_new(:)))./((max(image_new(:)))-min(image_new(:)))).*255;
% figure
% imshow(f)
% imwrite(f,new_file_name);
end
