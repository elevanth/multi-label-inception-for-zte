clc;clear all;close all;
% filename= 'xkdeconv.gif'; %你的gif文件的名字

files = dir('test\\*.jpg');
len = length(files);
for i=1:len
    tmp = files(i).name;
    str = ['test\\', tmp];
    img = imread(str);
    img_resize = imresize(img, [112,92]);
    img_rect_path = ['test_reshape\\', tmp];
    imwrite(img_resize,img_rect_path); 
end
close all;
