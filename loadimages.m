%Inputs: 
%foldername -- input string that says the folder location. Function
%assumes that folder is in cd, the same directory as your script
%view -- set to true / false depending if you want to view the folder's files
function [stack] = loadimages(foldername, view)
addpath('Z:\Documents')
%type dicomBrowser in command prompt to take a look
%VOLUME VIEWER APP DOESN'T WORK, RIP

dpath = fullfile(cd, foldername); %folder of images
ls1 = dicomCollection(dpath);
[V,s,d] = dicomreadVolume(ls1);
stack = squeeze(V); %now V is [rows,columns, slices]
if (view)
    figure, imshow3D(stack); %if you can't see anything, click Auto W/L
end

end