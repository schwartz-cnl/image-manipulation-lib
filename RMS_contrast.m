function [RMS] = RMS_contrast(I)
I = double(I);
mean_I = mean(I(:));
RMS = sqrt( sum((I(:)-mean_I).^2)/size(I(:),1) );
end

