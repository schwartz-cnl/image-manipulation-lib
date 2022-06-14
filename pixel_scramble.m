image_list = dir('images/original');
image_list = image_list(3:end);
% contrast = 76.5;
a = [];
for i = 1:size(image_list,1)
    image = imread(['images/original/' image_list(i).name]);
    a(i) = (get_alpha(squeeze(image(:,:,1)))+get_alpha(squeeze(image(:,:,1)))+get_alpha(squeeze(image(:,:,1))))/3;
end
mean_alpha = mean(a);

for i = 1:size(image_list,1)
    image = imread(['images/original/' image_list(i).name]);
    resized_image = imresize(image,[256 256]);
    alpha_im = resized_image;
%     alpha_im = change_slope_rgb(resized_image, mean_alpha);
    for scramble = [1,2,8,32,128]
        scrambled_image = pixel_scramble_im(alpha_im, scramble);
%         masked_scrambled_image = mask(scrambled_image);
        imwrite(uint8(abs(scrambled_image)), ['images/pixel_scramble/natural' int2str(i) '_scramble' int2str(scramble) '.png'])
    end
end

function scrambled_image = pixel_scramble_im(alpha_im, scramble)
    [w,~,c] = size(alpha_im);
    reshaped_image = zeros([scramble*scramble, w/scramble, w/scramble, c]);
    for i = 0:(scramble*scramble - 1)
        reshaped_image(i+1,:,:,:) = alpha_im((mod(i*w/scramble,w)+1):(mod(i*w/scramble,w)+w/scramble),...
                                    (floor(i/scramble)*w/scramble+1):(floor(i/scramble)*w/scramble+w/scramble),:);
    end
    p = randperm(scramble*scramble);
    scrambled_resahped_image = reshaped_image(p,:,:,:);
    scrambled_image = zeros(size(alpha_im));
    for i = 0:(scramble*scramble - 1)
        scrambled_image((mod(i*w/scramble,w)+1):(mod(i*w/scramble,w)+w/scramble),...
                                    (floor(i/scramble)*w/scramble+1):(floor(i/scramble)*w/scramble+w/scramble),:) = scrambled_resahped_image(i+1,:,:,:);
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

