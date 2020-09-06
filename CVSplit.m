function [nCV, dataIndsCV, dataIndsCVStarts, dataIndsCVEnds] = CVSplit(n, nCV)
% Cross-validation split

% nCV = round( n/1 )%10 %5 % LOOCV 

%         rand('state', 123); 

dataIndsCV = randperm(n);
dataIndsCVStarts = 1:round(n/nCV):(n-1); 
    if round(n/nCV) == 1
        dataIndsCVStarts = [dataIndsCVStarts, n];
    end        
if size(dataIndsCVStarts, 2) > nCV
    dataIndsCVStarts((nCV+1):end) = [];
end
dataIndsCVEnds = [dataIndsCVStarts(2:end) - 1, n];
nCV = size(dataIndsCVEnds, 2); % the true nCV
