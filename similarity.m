%get the images
rootFolder = 'E:\Images';
fullimds = imageDatastore(rootFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
imds = splitEachLabel(fullimds, 0.01, 'randomize');

%Load the network
cnnMatFile = fullfile('E:\get-the-look\Matlab', 'imagenet-caffe-alex.mat');
convnet = helperImportMatConvNet(cnnMatFile);

% The input layer requires 227x227x3 input => Set the ImageDatastore
% ReadFcn so that it reads and processes the image at the same time
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

% Extract features from the layer right before the classification layer.
featureLayer = 'fc7';

nrImagini = 1;
featureVectors = zeros(length(imds.Files),4096);
while hasdata(imds) > 0
    img = read(imds);
    featureVectors(nrImagini,:) = activations(convnet, img, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'rows'); 
    nrImagini = nrImagini + 1;
end
nrImagini = nrImagini - 1;

% Test image 
newImage = readAndPreprocessImage('E:\test9.jpg');
featureVector = activations(convnet, newImage, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'rows'); 

% Match the vectors using the Euclidean distance.

distanta = sum((repmat(featureVector,nrImagini,1) - featureVectors).^2,2);
[minim, index] = min(distanta);
figure,imshow(newImage);
figure,imshow(imread(imds.Files{index}));


