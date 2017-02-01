function Iout = readAndPreprocessImage(filename)
        I = imread(filename);
        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image.
        if ismatrix(I)
            I = cat(3,I,I,I);
        end
        % Resize the image as required for the CNN.
        % Get the aspect ratio
        H = size(I,1);
        W = size(I,2);
        k = H/W;
        % Set the smallest dimension equal to 227
        % Compute the other keeping the aspect ratio
        % Classic resize
        % Content-resize so that the largest dimension is equal to 227
        Iout = uint8(zeros(227,227,3));
        if H > W
            H = 227;
            W = floor(H/k);
            Iresized = imresize(I, [H W]);
            Iout(:,1:floor((227-W)/2),:) = repmat([255],H,floor((227-W)/2),3);
            Iout(:,floor((227-W)/2)+1:floor((227-W)/2)+W,:) = Iresized;
            Iout(:,floor((227-W)/2)+W+1:227,:) = repmat([255],H,227-floor((227-W)/2)-W,3);
        else
            Iout = imresize(I,[227 227]);
        end 
    end