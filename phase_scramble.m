image_list = dir('images/original');
image_list = image_list(3:end);
a = [];
contrast = 76.5;
% for i = 1:size(image_list,1)
%     image = imread(['images/original/' image_list(i).name]);
%     a(i) = (get_alpha(squeeze(image(:,:,1)))+get_alpha(squeeze(image(:,:,1)))+get_alpha(squeeze(image(:,:,1))))/3;
% %     for alpha = 0:3
% %         alpha_im = change_slope_rgb(image, alpha);
% %         imwrite(alpha_im, ['images/color_only_change_alpha/natural' int2str(i) '_a' int2str(alpha) '.png'])
% %     end
% end
% mean_alpha = mean(a);

for i = 1:size(image_list,1)
    image = imread(['images/original/' image_list(i).name]);
    resized_image = imresize(image,[256 256]);
%     alpha_im = change_slope_rgb(resized_image, mean_alpha);
    alpha_im = resized_image;
    rms_contrast_r = RMS_contrast(alpha_im(:,:,1));
    rms_contrast_g = RMS_contrast(alpha_im(:,:,2));
    rms_contrast_b = RMS_contrast(alpha_im(:,:,3));
    rms_contrast = sqrt(rms_contrast_r^2+rms_contrast_g^2+rms_contrast_b^2);
    alpha_im = double(alpha_im)/rms_contrast*contrast + 128*(1-contrast/rms_contrast);
    
    for phase_coherence = [0, 25, 50, 75, 100]
        random_phase = random('unif',-pi,pi,[size(alpha_im), 3]);
        noise_image = phase_scramble_im(alpha_im, random_phase, phase_coherence*0.01);
%         noise_image = mask(noise_image);
        imwrite(uint8(abs(noise_image)), ['images/phase_scramble/natural' int2str(i) '_coherence' int2str(phase_coherence) '.png'])
    end
end

function noise_image = phase_scramble_im(alpha_im, random_phase, phase_coherence)
    for channel = 1:3
        fft2_im(:,:,channel) = fft2(squeeze(alpha_im(:,:,channel)));
        fft2_im_phs(:,:,channel) = angle(fft2_im(:,:,channel));
        fft2_im_amp(:,:,channel) = abs(fft2_im(:,:,channel));
    end
    new_phs = phase_coherence*fft2_im_phs + (1-phase_coherence)*random_phase;
    [x,y] = pol2cart(new_phs,fft2_im_amp);
    noise_image_fft2 = x + 1i*y; 
    for channel = 1:3
        noise_image(:,:,channel) = ifft2(squeeze(noise_image_fft2(:,:,channel)));
    end
end

function masked_image = mask(image)
    [w,~,~] = size(image);
    masked_image = image;
    gray = ones(size(image))*128;
    filter1(:,1,1) = (0:24)/24;
    filter1 = repmat(filter1, 1,w,3);
    filter2(1,:,1) = (0:24)/24;
    filter2 = repmat(filter2, w,1,3);
    masked_image(1:25,:,:) = masked_image(1:25,:,:) .* filter1 + gray(1:25,:,:) .* (1-filter1);
    masked_image(256:-1:232,:,:) = masked_image(256:-1:232,:,:) .* filter1 + gray(256:-1:232,:,:) .* (1-filter1);
    masked_image(:,1:25,:) = masked_image(:,1:25,:) .* filter2 + gray(:,1:25,:) .* (1-filter2);
    masked_image(:,256:-1:232,:) = masked_image(:,256:-1:232,:) .* filter2 + gray(:,256:-1:232,:) .* (1-filter2);
end