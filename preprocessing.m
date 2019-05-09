% Main function for preprocessing
function [] = preprocessing()

stack1 = loadimages('128_10_27_15', true); %load example folder of images
stack1 = stack1/(max(max(max(stack1)))-min(min(min(stack1)))); %normalizes it

end
