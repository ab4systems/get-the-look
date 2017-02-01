function featureVector = getFeatureVector(filepath)

    %Load the network
    cnnMatFile = fullfile('E:\get-the-look\Matlab', 'imagenet-caffe-alex.mat');
    convnet = helperImportMatConvNet(cnnMatFile);
    
    %Load image 
    newImage = readAndPreprocessImage(filepath);
    featureLayer = 'fc7';
    featureVector = activations(convnet, newImage, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'rows'); 
end

