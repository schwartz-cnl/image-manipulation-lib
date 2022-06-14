% color natural images, change power spectrum and contrast.
image_list = dir('images/original');
image_list = image_list(3:end);
contrast = 76.5;
for i = 1:size(image_list,1)
    image = imread(['images/original/' image_list(i).name]);
    image = imresize(image,[256 256]);
%     rms_contrast = RMS_contrast(rgb2gray(image));
    rms_contrast_r = RMS_contrast(image(:,:,1));
    rms_contrast_g = RMS_contrast(image(:,:,2));
    rms_contrast_b = RMS_contrast(image(:,:,3));
    rms_contrast = sqrt(rms_contrast_r^2+rms_contrast_g^2+rms_contrast_b^2);
    contrast_im = double(image)/rms_contrast*contrast + 128*(1-contrast/rms_contrast);
    for noise = [0, 30, 60, 90, 120, 135, 150, 180]
        noise_image = add_phase_noise(contrast_im,noise/180*pi);
%         moise_image = mask(noise_image);
        imwrite(uint8(noise_image), ['images/rieger2013/natural' int2str(i) '_phasenoise' int2str(noise) '.png'])
    end
%     rms_contrast_r = RMS_contrast(noise_image(:,:,1));
%     rms_contrast_g = RMS_contrast(noise_image(:,:,2));
%     rms_contrast_b = RMS_contrast(noise_image(:,:,3));
%     rms_contrast = sqrt(rms_contrast_r^2+rms_contrast_g^2+rms_contrast_b^2)
% The above commented part is for checking rms contrast is changed
% expected, after changing contrast and adding noise.
end

%%
% Kurtosis
image_list = dir('images/original');
image_list = image_list(3:end);
total_k = zeros(100,8);
for i = 1:size(image_list,1)
    im_k = [];
    for noise = [0, 30, 60, 90, 120, 135, 150, 180]
        image = imread(['images/rieger2013/natural' int2str(i) '_phasenoise' int2str(noise) '.png']);
        k = [];
        for c = 1:3
            single_c_im = double(image(:,:,c));
            single_c_kurtosis = kurtosis(single_c_im(:));
            if single_c_kurtosis<20 % some image has constant channel.
                k = [k, single_c_kurtosis];
            end
        end
        im_k = [im_k, mean(k)];
    end
    total_k(i,:) = im_k;
end
%%
% phase-only kurtosis
image_list = dir('images/original');
image_list = image_list(3:end);
total_k = zeros(100,8);
for i = 1:size(image_list,1)
    im_k = [];
    for noise = [0, 30, 60, 90, 120, 135, 150, 180]
        image = imread(['images/rieger2013/natural' int2str(i) '_phasenoise' int2str(noise) '.png']);
        rms_contrast_r = RMS_contrast(image(:,:,1));
        rms_contrast_g = RMS_contrast(image(:,:,2));
        rms_contrast_b = RMS_contrast(image(:,:,3));
        rms_contrast_ori = sqrt(rms_contrast_r^2+rms_contrast_g^2+rms_contrast_b^2);
        alpha_im = change_slope_rgb(double(image), 0);
        rms_contrast_r =    RMS_contrast(alpha_im(:,:,1));
        rms_contrast_g = RMS_contrast(alpha_im(:,:,2));
        rms_contrast_b = RMS_contrast(alpha_im(:,:,3));
        rms_contrast = sqrt(rms_contrast_r^2+rms_contrast_g^2+rms_contrast_b^2);
        image = single(alpha_im)/rms_contrast*rms_contrast_ori + 128*(1-rms_contrast_ori/rms_contrast);
        k = [];
        for c = 1:3
            single_c_im = double(image(:,:,c));
            single_c_kurtosis = kurtosis(single_c_im(:));
            if single_c_kurtosis<50 % some image has constant channel.
                k = [k, single_c_kurtosis];
            end
        end
        im_k = [im_k, mean(k)];
    end
    total_k(i,:) = im_k;
end
%%
% phase-only kurtosis differences
image_list = dir('images/original');
image_list = image_list(3:end);
total_k = zeros(100,8);
for i = 1:size(image_list,1)
    im_k = [];
    for noise = [0, 30, 60, 90, 120, 135, 150, 180]
        image = imread(['images/rieger2013/natural' int2str(i) '_phasenoise' int2str(noise) '.png']);
        phase_perm_image = phase_permute(image);
        alpha_im = change_slope_rgb(double(image), 0);
        alpha_phase_perm_image = change_slope_rgb(double(phase_perm_image), 0);
        k = [];
        for c = 1:3
            k(c) = size(alpha_im,1)*size(alpha_im,2) * abs(sum(sum(alpha_phase_perm_image(:,:,c).^4))-...
            sum(sum(alpha_im(:,:,c).^4))) / (sum(sum(alpha_im(:,:,c).^2)))^2;
        end
        im_k = [im_k, mean(k)];
    end
    total_k(i,:) = im_k;
end
%%
% phase-only kurtosis (blending)
image_list = dir('images/original');
image_list = image_list(3:end);
total_k = zeros(100,5);
for i = 1:size(image_list,1)
    im_k = [];
    for phase_coherence = [0, 25, 50, 75, 100]
        image = imread(['images/phase_scramble/natural' int2str(i) '_coherence' int2str(phase_coherence) '.png']);
        rms_contrast_r = RMS_contrast(image(:,:,1));
        rms_contrast_g = RMS_contrast(image(:,:,2));
        rms_contrast_b = RMS_contrast(image(:,:,3));
        rms_contrast_ori = sqrt(rms_contrast_r^2+rms_contrast_g^2+rms_contrast_b^2);
        alpha_im = change_slope_rgb(double(image), 0);
        rms_contrast_r = RMS_contrast(alpha_im(:,:,1));na
        rms_contrast_g = RMS_contrast(alpha_im(:,:,2));
        rms_contrast_b = RMS_contrast(alpha_im(:,:,3));
        rms_contrast = sqrt(rms_contrast_r^2+rms_contrast_g^2+rms_contrast_b^2);
        image = single(alpha_im)/rms_contrast*rms_contrast_ori + 128*(1-rms_contrast_ori/rms_contrast);
        k = [];
        for c = 1:3
            single_c_im = double(image(:,:,c));
            single_c_kurtosis = kurtosis(single_c_im(:));
            if single_c_kurtosis<50 % some image has constant channel.
                k = [k, single_c_kurtosis];
            end
        end
        im_k = [im_k, mean(k)];
    end
    total_k(i,:) = im_k;
end
%%
% phase-only kurtosis differences (blending)
image_list = dir('images/original');
image_list = image_list(3:end);
total_k = zeros(100,5);
for i = 1:size(image_list,1)
    im_k = [];
    for phase_coherence = [0, 25, 50, 75, 100]
        image = imread(['images/phase_scramble/natural' int2str(i) '_coherence' int2str(phase_coherence) '.png']);
        image = double(rgb2gray(image))-128;
        phase_perm_image = phase_permute_gray(image);
        alpha_im = change_slope_im(double(image), 0);
        alpha_phase_perm_image = change_slope_im(double(phase_perm_image), 0);
%         phase_perm_image = phase_permute(image);
%         alpha_im = change_slope_rgb(double(image), 0);
%         alpha_phase_perm_image = change_slope_rgb(double(phase_perm_image), 0);
        k = [];
        for c = 1:1
            single_c_kurtosis = size(alpha_im,1)*size(alpha_im,2) * abs(sum(sum(alpha_phase_perm_image(:,:,c).^4))-...
            sum(sum(alpha_im(:,:,c).^4))) / (sum(sum(alpha_im(:,:,c).^2)))^2;
            if single_c_kurtosis<10 % some image has constant channel.
                k = [k, single_c_kurtosis];
            end
        end
        im_k = [im_k, mean(k)];
    end
    total_k(i,:) = im_k;
end
%%
% RMS contrast
image_list = dir('images/original');
image_list = image_list(3:end);
total_RMS = zeros(100,8);
for i = 1:size(image_list,1)
    im_RMS = [];
    for noise = [0, 30, 60, 90, 120, 135, 150, 180]
        image = imread(['images/rieger2013/natural' int2str(i) '_phasenoise' int2str(noise) '.png']);
        RMS = [];
        for c = 1:3
            single_c_im = double(image(:,:,c));
            single_c_RMS = RMS_contrast(single_c_im(:));
            RMS = [RMS, single_c_RMS];
        end
        im_RMS = [im_RMS, rms(RMS)];
    end
    total_RMS(i,:) = im_RMS;
end

%%
function noise_image = add_phase_noise(im, max_angle)
    [h, w, ~] = size(im);
    for channel = 1:3
        fft2_im(:,:,channel) = fft2(squeeze(im(:,:,channel)));
        fft2_im_phs(:,:,channel) = angle(fft2_im(:,:,channel));
        fft2_im_amp(:,:,channel) = abs(fft2_im(:,:,channel));
    end
    % center
    phase = fft2_im_phs(1,1,:)+(rand(1,1,3)*2*max_angle-max_angle);
    [x, y] = pol2cart(phase,fft2_im_amp(1,1,:));
    noise_image_fft2(1,1,:) = x +1i*y;
    % x = 0
    phase = fft2_im_phs(1,ceil(w/2)+1:end,:)+(rand(1,floor(w/2),3)*2*max_angle-max_angle);
    [x, y] = pol2cart(phase,fft2_im_amp(1,ceil(w/2)+1:end,:));
    noise_image_fft2(1,ceil(w/2)+1:w,:) = x + 1i*y;
    noise_image_fft2(1,2:ceil(w/2),:) = x(1,end:-1:2-mod(w,2),:) - 1i*y(1,end:-1:2-mod(w,2),:);
    % y = 0
    phase = fft2_im_phs(ceil(h/2)+1:end,1,:)+(rand(floor(h/2),1,3)*2*max_angle-max_angle);
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+1:end,1,:));
    noise_image_fft2(ceil(h/2)+1:h,1,:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2),1,:) = x(end:-1:2-mod(h,2),1,:) - 1i*y(end:-1:2-mod(h,2),1,:);
    % quadrant 2,4
    phase = fft2_im_phs(ceil(h/2)+1:end,ceil(w/2)+1:end,:)+(rand(floor(h/2),floor(w/2),3)*2*max_angle-max_angle);
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+1:end,ceil(w/2)+1:end,:));
    noise_image_fft2(ceil(h/2)+1:h,ceil(w/2)+1:w,:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2)+mod(h+1,2),2:ceil(w/2)+mod(w+1,2),:) = x(end:-1:1,end:-1:1,:) - 1i*y(end:-1:1,end:-1:1,:);
    noise_image_fft2(ceil(h/2)+1,ceil(w/2)+1,:) = x(1,1,:) + 1i*y(1,1,:);
    % quadrant 1,3
    phase = fft2_im_phs(ceil(h/2)+2:end,2:ceil(w/2),:)+(rand(floor(h/2)-1,ceil(w/2)-1,3)*2*max_angle-max_angle);
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+2:end,2:ceil(w/2),:));
    noise_image_fft2(ceil(h/2)+2:h,2:ceil(w/2),:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2)-mod(h,2),ceil(w/2)+2-mod(w,2):w,:) = x(end:-1:1,end:-1:1,:) - 1i*y(end:-1:1,end:-1:1,:);
    % fix zero phase points
    noise_image_fft2(1,1,:) = fft2_im(1,1,:);
    if mod(h,2)==0
        noise_image_fft2(ceil(h/2)+1,1,:) = fft2_im(ceil(h/2)+1,1,:);
    end
    if mod(w,2)==0
        noise_image_fft2(1,ceil(w/2)+1,:) = fft2_im(1,ceil(w/2)+1,:);
    end
    if mod(h,2)==0 && mod(w,2)==0
        noise_image_fft2(ceil(h/2)+1,ceil(w/2)+1,:) = fft2_im(ceil(h/2)+1,ceil(w/2)+1,:);
    end

    for channel = 1:3
        noise_image(:,:,channel) = ifft2(squeeze(noise_image_fft2(:,:,channel)));
    end
end

function masked_image = mask(image)
    mask_size = 10;
    [w,~,~] = size(image);
    masked_image = image;
    gray = ones(size(image))*128;
    filter1(:,1,1) = (0:mask_size-1)/(mask_size-1);
    filter1 = repmat(filter1, 1,w,3);
    filter2(1,:,1) = (0:mask_size-1)/(mask_size-1);
    filter2 = repmat(filter2, w,1,3);
    masked_image(1:mask_size,:,:) = masked_image(1:mask_size,:,:) .* filter1 + gray(1:mask_size,:,:) .* (1-filter1);
    masked_image(256:-1:257-mask_size,:,:) = masked_image(256:-1:257-mask_size,:,:) .* filter1 + gray(256:-1:257-mask_size,:,:) .* (1-filter1);
    masked_image(:,1:mask_size,:) = masked_image(:,1:mask_size,:) .* filter2 + gray(:,1:mask_size,:) .* (1-filter2);
    masked_image(:,256:-1:257-mask_size,:) = masked_image(:,256:-1:257-mask_size,:) .* filter2 + gray(:,256:-1:257-mask_size,:) .* (1-filter2);
end

function phase_permute_im = phase_permute(im)
    im = double(im);
    [h, w, ~] = size(im);
    for channel = 1:3
        fft2_im(:,:,channel) = fft2(squeeze(im(:,:,channel)));
        fft2_im_phs(:,:,channel) = angle(fft2_im(:,:,channel));
        fft2_im_amp(:,:,channel) = abs(fft2_im(:,:,channel));
    end
    % center
    phase = fft2_im_phs(1,1,:);
    [x, y] = pol2cart(phase,fft2_im_amp(1,1,:));
    noise_image_fft2(1,1,:) = x +1i*y;
    % x = 0
    temp = fft2_im_phs(1,ceil(w/2)+1:end,:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(1,ceil(w/2)+1:end,:));
    noise_image_fft2(1,ceil(w/2)+1:w,:) = x + 1i*y;
    noise_image_fft2(1,2:ceil(w/2),:) = x(1,end:-1:2-mod(w,2),:) - 1i*y(1,end:-1:2-mod(w,2),:);
    % y = 0
    temp = fft2_im_phs(ceil(h/2)+1:end,1,:,:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+1:end,1,:));
    noise_image_fft2(ceil(h/2)+1:h,1,:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2),1,:) = x(end:-1:2-mod(h,2),1,:) - 1i*y(end:-1:2-mod(h,2),1,:);
    % quadrant 2,4
    temp = fft2_im_phs(ceil(h/2)+1:end,ceil(w/2)+1:end,:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+1:end,ceil(w/2)+1:end,:));
    noise_image_fft2(ceil(h/2)+1:h,ceil(w/2)+1:w,:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2)+mod(h+1,2),2:ceil(w/2)+mod(w+1,2),:) = x(end:-1:1,end:-1:1,:) - 1i*y(end:-1:1,end:-1:1,:);
    noise_image_fft2(ceil(h/2)+1,ceil(w/2)+1,:) = x(1,1,:) + 1i*y(1,1,:);
    % quadrant 1,3
    temp = fft2_im_phs(ceil(h/2)+2:end,2:ceil(w/2),:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+2:end,2:ceil(w/2),:));
    noise_image_fft2(ceil(h/2)+2:h,2:ceil(w/2),:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2)-mod(h,2),ceil(w/2)+2-mod(w,2):w,:) = x(end:-1:1,end:-1:1,:) - 1i*y(end:-1:1,end:-1:1,:);
    % fix zero phase points
    noise_image_fft2(1,1,:) = fft2_im(1,1,:);
    if mod(h,2)==0
        noise_image_fft2(ceil(h/2)+1,1,:) = fft2_im(ceil(h/2)+1,1,:);
    end
    if mod(w,2)==0
        noise_image_fft2(1,ceil(w/2)+1,:) = fft2_im(1,ceil(w/2)+1,:);
    end
    if mod(h,2)==0 && mod(w,2)==0
        noise_image_fft2(ceil(h/2)+1,ceil(w/2)+1,:) = fft2_im(ceil(h/2)+1,ceil(w/2)+1,:);
    end

    for channel = 1:3
        phase_permute_im(:,:,channel) = ifft2(squeeze(noise_image_fft2(:,:,channel)));
    end
end

function phase_permute_im = phase_permute_gray(im)
    im = double(im);
    [h, w] = size(im);
    im(:,:,1) = im;
    for channel = 1:1
        fft2_im(:,:,channel) = fft2(squeeze(im(:,:,channel)));
        fft2_im_phs(:,:,channel) = angle(fft2_im(:,:,channel));
        fft2_im_amp(:,:,channel) = abs(fft2_im(:,:,channel));
    end
    % center
    phase = fft2_im_phs(1,1,:);
    [x, y] = pol2cart(phase,fft2_im_amp(1,1,:));
    noise_image_fft2(1,1,:) = x +1i*y;
    % x = 0
    temp = fft2_im_phs(1,ceil(w/2)+1:end,:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(1,ceil(w/2)+1:end,:));
    noise_image_fft2(1,ceil(w/2)+1:w,:) = x + 1i*y;
    noise_image_fft2(1,2:ceil(w/2),:) = x(1,end:-1:2-mod(w,2),:) - 1i*y(1,end:-1:2-mod(w,2),:);
    % y = 0
    temp = fft2_im_phs(ceil(h/2)+1:end,1,:,:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+1:end,1,:));
    noise_image_fft2(ceil(h/2)+1:h,1,:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2),1,:) = x(end:-1:2-mod(h,2),1,:) - 1i*y(end:-1:2-mod(h,2),1,:);
    % quadrant 2,4
    temp = fft2_im_phs(ceil(h/2)+1:end,ceil(w/2)+1:end,:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+1:end,ceil(w/2)+1:end,:));
    noise_image_fft2(ceil(h/2)+1:h,ceil(w/2)+1:w,:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2)+mod(h+1,2),2:ceil(w/2)+mod(w+1,2),:) = x(end:-1:1,end:-1:1,:) - 1i*y(end:-1:1,end:-1:1,:);
    noise_image_fft2(ceil(h/2)+1,ceil(w/2)+1,:) = x(1,1,:) + 1i*y(1,1,:);
    % quadrant 1,3
    temp = fft2_im_phs(ceil(h/2)+2:end,2:ceil(w/2),:);
    temp = temp(:,randperm(size(temp,2)),:);
    temp = temp(randperm(size(temp,1)),:,:);
    phase = temp;
    [x, y] = pol2cart(phase,fft2_im_amp(ceil(h/2)+2:end,2:ceil(w/2),:));
    noise_image_fft2(ceil(h/2)+2:h,2:ceil(w/2),:) = x + 1i*y;
    noise_image_fft2(2:ceil(h/2)-mod(h,2),ceil(w/2)+2-mod(w,2):w,:) = x(end:-1:1,end:-1:1,:) - 1i*y(end:-1:1,end:-1:1,:);
    % fix zero phase points
    noise_image_fft2(1,1,:) = fft2_im(1,1,:);
    if mod(h,2)==0
        noise_image_fft2(ceil(h/2)+1,1,:) = fft2_im(ceil(h/2)+1,1,:);
    end
    if mod(w,2)==0
        noise_image_fft2(1,ceil(w/2)+1,:) = fft2_im(1,ceil(w/2)+1,:);
    end
    if mod(h,2)==0 && mod(w,2)==0
        noise_image_fft2(ceil(h/2)+1,ceil(w/2)+1,:) = fft2_im(ceil(h/2)+1,ceil(w/2)+1,:);
    end

    for channel = 1:1
        phase_permute_im(:,:,channel) = ifft2(squeeze(noise_image_fft2(:,:,channel)));
    end
    phase_permute_im = squeeze(phase_permute_im);
end