 function [B] = Func_Thresholding(A, thresholdingWay, Lambda);
% This function performs thresholding operations on a vector or matrix
% lambda is of the same size of A
B = zeros(size(A));
switch lower(thresholdingWay)
    case 'hard'
        tmpInds1 = find( abs(A) >= Lambda ); 
        B(tmpInds1) = A(tmpInds1);
    case 'soft'
        tmpInds1 = find( A >= Lambda ); 
        tmpInds2 = find( A <= -Lambda ); 
        B(tmpInds1) = A(tmpInds1) - Lambda(tmpInds1);
        B(tmpInds2) = A(tmpInds2) + Lambda(tmpInds2);
    case 'soft_nonnegative'
        tmpInds1 = find( A >= Lambda ); 
        B(tmpInds1) = A(tmpInds1) - Lambda(tmpInds1);
    case 'berhu'
        berhu_eta = 1e-7;
        tmpInds1 = find( A >= Lambda & A <= Lambda + Lambda / berhu_eta ); 
        tmpInds2 = find( A <= -Lambda & A >= -Lambda - Lambda / berhu_eta ); 
        tmpInds3 = find ( A >= Lambda + Lambda / berhu_eta);
        tmpInds4 = find ( A <= -Lambda -Lambda / berhu_eta);
        B(tmpInds1) = A(tmpInds1) - Lambda(tmpInds1);
        B(tmpInds2) = A(tmpInds2) + Lambda(tmpInds2);
        B(tmpInds3) = A(tmpInds3) / (1 + berhu_eta);
        B(tmpInds4) = A(tmpInds4) / (1 + berhu_eta);

    otherwise
        error('Not implemented yet');
end
 end