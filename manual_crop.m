
clc;clear all;close all;

files = dir('database\\*.jpg');
len = length(files);
for i=1:len
    tmp = files(i).name;
    str = ['database\\', tmp];
    img = imread(str);
    crop = imcrop(img);
    [hh,ww] = size(crop);
        if hh <= 90
            fprintf('%s img not cropped',str);
            continue;
        else
            crop_resize = imresize(crop, [112,92]);
            img_rect_path = ['cropped\\', 'add', int2str(i), '.jpg'];
            imwrite(crop_resize,img_rect_path); 
        end
end
close all;
