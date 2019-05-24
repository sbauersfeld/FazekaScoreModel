% Main function for preprocessing
function [] = preprocessing()
fnames = ["122_10_20_15", "128_10_27_15"];
% fnames = ['128_10_27_15'];
for f = 1:length(fnames)
    stack = loadimages(fnames(f), false); %load folder of images
    stack = stack/(max(max(max(stack)))-min(min(min(stack))));
    save("p"+fnames(f)+".mat", 'stack');
end

end
