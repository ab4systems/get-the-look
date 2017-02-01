%get the images
rootFolder = 'E:\Images';
imds = imageDatastore(rootFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
tbl = countEachLabel(imds);

minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Location of pre-trained "AlexNet"
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-caffe-alex.mat';
% Store CNN model in a folder
cnnMatFile = fullfile('E:\get-the-look\Matlab', 'imagenet-caffe-alex.mat');
if ~exist(cnnMatFile, 'file') % download only once
    disp('Downloading pre-trained CNN model...');
    websave(cnnMatFile, cnnURL);
end

%Load the network
convnet = helperImportMatConvNet(cnnMatFile)

% The input layer requires 227x227x3 input => Set the ImageDatastore
% ReadFcn so that it reads and processes the image at the same time
imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

% Split the sets into training and validation data. Pick 30% of images from 
% each set for the training data and the remainder, 70%, for the validation 
% data. Randomize the split to avoid biasing the results. The training and 
% test sets will be processed by the CNN model.
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize');

% Extract features from the layer right before the classification layer.
featureLayer = 'fc7';
trainingFeatures = activations(convnet, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

% Next, using the CNN image features, we train a multiclass SVM classifier. 
% A fast Stochastic Gradient Descent solver is used for training by setting 
% the fitcecoc function's 'Learners' parameter to 'Linear'. This helps 
% speed-up the training when working with high-dimensional CNN feature 
% vectors, which each have a length of 4096. Set 'ObservationsIn' to 
% 'columns' to match the arrangement used for training features.
classifier = fitcecoc(trainingFeatures, trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Evaluate Classifier

% Extract test features using the CNN
testFeatures = activations(convnet, testSet, featureLayer, 'MiniBatchSize',32);

% Pass CNN image features to trained classifier
predictedLabels = predict(classifier, testFeatures);

% Get the known labels
testLabels = testSet.Labels;

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

% Display the mean accuracy
mean(diag(confMat))

%Save the network

save('svm.mat','convnet');

% Try the Newly Trained Classifier on Test Images

newImage = fullfile(rootFolder, 'Men-Bags', '100.jpg');

% Pre-process the images as required for the CNN
img = readAndPreprocessImage(newImage);

% Extract image features using the CNN
imageFeatures = activations(convnet, img, featureLayer);

% Make a prediction using the classifier
label = predict(classifier, imageFeatures)

