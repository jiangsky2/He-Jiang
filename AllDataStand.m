function [X, valX, tstX, trnCenters, trnScales, y, valY, tstY, trnCenterY] = AllDataStand(centerX, scaleX, X, valX, tstX, centerY, y, valY, tstY)
% Use the parameters obtained from the training data to standardize the
% data (including the validation and the test data) for both the design and
% the response. 
trnCenters = []; trnScales = []; trnCenterY = [];
if centerX == 1 % X should not include the intercept column
    % Note y-centering is not appropriate for nonGaussian GLM (change the distr)
    trnCenters = mean(X);
    X = X - repmat(trnCenters, size(X, 1), 1);
    if ~isempty(valX)
        valX = valX - repmat(trnCenters, size(valX, 1), 1);
    end
    if ~isempty(tstX)
        tstX = tstX - repmat(trnCenters, size(tstX, 1), 1);
    end
end
if scaleX == 1 % Assume X does not include the intercept column in this case
    % Col-normalize X to have all L2-norms equal to sqrt(n)/sqrt(valN)/sqrt(tstN)
    % Use the same params to scale val & tst  
    trnScales = sqrt(sum(X.^2)/size(X, 1));
    X = X .* repmat(1./trnScales, size(X, 1), 1); % X * diag(1./(trnScales));
    if ~isempty(valX)
        valX = valX .* repmat(1./trnScales, size(valX, 1), 1);
    end
    if ~isempty(tstX)
        tstX = tstX .* repmat(1./trnScales, size(tstX, 1), 1);
    end
end
if centerY == 1  
    % Use the same(!) params to center valY and tstY
    
    trnCenterY = mean(y);
    
%     y = y - trnCenterY; %univariate y
    y = y - repmat(trnCenterY, size(y, 1), 1); 
    if ~isempty(valY)
%         valY = valY - trnCenterY; %univariate y
        valY = valY - repmat(trnCenterY, size(valY, 1), 1); 
    end
    if ~isempty(tstY)
%         tstY = tstY - trnCenterY; %univariate y
        tstY = tstY - repmat(trnCenterY, size(tstY, 1), 1); 
    end
end