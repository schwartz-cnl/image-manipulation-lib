function f = change_slope_rgb(image, alpha)
% function f=change_slope_rgb(original_image, alpha, square, new_file_name)

% image = imread(original_image);
size_image = size(image);
L = min(size_image(1:2));
% if square==0
    image = imresize(image, [L L], 'bicubic');
% end
image_r = image(:,:,1);
image_g = image(:,:,2);
image_b = image(:,:,3);

f(:,:,1) = change_slope_channel(image_r,alpha);
f(:,:,2) = change_slope_channel(image_g,alpha);
f(:,:,3) = change_slope_channel(image_b,alpha);

% figure
% imshow(f)
% imwrite(f,new_file_name);
end


function [f] = change_slope_channel(image,alpha)
    scale = range(image(:));
    size_image = size(image);
    L = min(size_image(1:2));
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
    f = ((image_new-min(image_new(:)))./((max(image_new(:)))-min(image_new(:)))).*double(scale);
%     f = uint8(((image_new-min(image_new(:)))./((max(image_new(:)))-min(image_new(:)))).*double(scale));
end