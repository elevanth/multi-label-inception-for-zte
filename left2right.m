function left2right( str )
%LEFT2RIGHT Summary of this function goes here
%   Detailed explanation goes here
    real_path = strcat(str, '\\', '*.jpg');
    files = dir(real_path);
    len = length(files);
    for i=1:len
        s = fullfile(str, '\\', files(i).name);
        img = imread(s);
        l2r = img(:, end:-1:1, 1:3);
        
        newname = strcat(str, '\\', 'l2r_', files(i).name);
        imwrite(l2r, newname);
    end
    close all
end

