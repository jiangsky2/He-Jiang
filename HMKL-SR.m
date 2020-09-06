clear,  clc;
resoutput = false % false % 
randn('state', 0); rand('state', 0);

% Generate your data with X and y
X
y

[n,p] = size(X);


dataName = ;

% Generate NumP feature representations 
NumP = 11;
subset = cell(1, NumP);

% The number of features in one representation
NumFeatureInSubset =      ; 


for j = 1 : NumP
    index0 = randperm(p);
     index1 = index0(1:NumFeatureInSubset);
    subset{j} = X(:,index1);
end



% Generate kernels
run KernelFile.m

% Initial weights paraemter 
alpha_0 = 1 / NumP * ones(NumP, 1);
H_org = [H_LK H_RBF1 H_RBF2 H_RBF3 H_RBF4 H_RBF5 H_RBF6 H_RBF7 H_RBF8 H_RBF9 H_RBF10];

% CV for evaluating prediction accuracy
outputDisp = true 
resoutput = true
dataoutput = false 
matlaboutput = true
if resoutput, diary(['matlaboutput', dataName, '-starttime-', datestr(clock, '-yyyy-mm-dd-HH.MM.SS'), '.txt']), end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
multiSplit = 0 
if multiSplit == 0
    nCV_outer = 5; 
elseif multiSplit == 1
    splittingTimes = 100 
    tstRatio = 0.1
end
ALLDATA = [y, H_org];  numResps = 1;
n = size(ALLDATA, 1);
d = size(H_org, 2);

if multiSplit == 0
    [nCV_outer, dataIndsCV_outer, dataIndsCVStarts_outer, dataIndsCVEnds_outer] = CVSplit(size(ALLDATA, 1), nCV_outer);
    splittingTimes = nCV_outer;
end
if multiSplit == 2
    splittingTimes = 1;
end
tsterrs = zeros(splittingTimes, 1);
numNzOpts = zeros(splittingTimes, 1);
AllpredOpts = cell(splittingTimes, 1);
trnInds_CV = cell(splittingTimes, 1);
tstInds_CV = cell(splittingTimes, 1);

  
for splitTimeInd = 1:splittingTimes
    if resoutput, diary on, end
    disp(['splitTimeInd=', num2str(splitTimeInd)])
    if multiSplit == 0    % perform CV
        tstInds = dataIndsCV_outer( dataIndsCVStarts_outer(splitTimeInd):dataIndsCVEnds_outer(splitTimeInd) );
        trnInds = 1:size(ALLDATA, 1); trnInds(tstInds) = [];
    elseif multiSplit == 1             % perform multi-split
        trnInds = randsample(n, round((1-tstRatio) * n));
        tstInds = 1:size(ALLDATA, 1); tstInds(trnInds) = [];
    elseif multiSplit == 2
        tstInds = sepTestInds;
        trnInds = 1:size(ALLDATA, 1); trnInds(tstInds) = [];
    else
        error('Wrong multiSplit code!')
    end
    H_trn0{splitTimeInd} = ALLDATA(trnInds, (1+numResps):end); y_trn0{splitTimeInd} = ALLDATA(trnInds, 1:numResps); 
    H_tst0{splitTimeInd} = ALLDATA(tstInds, (1+numResps):end); y_tst0{splitTimeInd} = ALLDATA(tstInds, 1:numResps); 
        
    trnInds_CV{splitTimeInd} = trnInds;   tstInds_CV{splitTimeInd} = tstInds;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Training  Procedure%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 for   splitTimeInd = 1:splittingTimes
   
% Generate training and test datasets     
H_trn = H_trn0{splitTimeInd};
H_tst = H_tst0{splitTimeInd};
y_trn = y_trn0{splitTimeInd};
y_tst = y_tst0{splitTimeInd};
trnInds = trnInds_CV{splitTimeInd};
tstInds = tstInds_CV{splitTimeInd};
      
% Scaling of training and test datasets 
[H_trn_c, ~,H_tst_c, trnCenters, trnScales, y_trn_c, ~, y_tst_c, trnCenterY] = AllDataStand(1, 1, H_trn, [], H_tst , 1, y_trn, [], y_tst);

% Get H martix for different kernels
H_LK = H_trn_c(:,1:SVD_cutoffPoint);
H_RBF1 = H_trn_c(:,SVD_cutoffPoint + 1: 2 * SVD_cutoffPoint);
H_RBF2 = H_trn_c(:,2 * SVD_cutoffPoint + 1: 3 * SVD_cutoffPoint);
H_RBF3 = H_trn_c(:,3 * SVD_cutoffPoint + 1: 4 * SVD_cutoffPoint);
H_RBF4 = H_trn_c(:,4 * SVD_cutoffPoint + 1: 5 * SVD_cutoffPoint);
H_RBF5 =H_trn_c(:,5 * SVD_cutoffPoint + 1: 6 * SVD_cutoffPoint);
H_RBF6 = H_trn_c(:,6 * SVD_cutoffPoint + 1: 7 * SVD_cutoffPoint);
H_RBF7 =H_trn_c(:, 7 * SVD_cutoffPoint + 1: 8 * SVD_cutoffPoint);
H_RBF8 =H_trn_c(:,8 * SVD_cutoffPoint + 1: 9 * SVD_cutoffPoint);
H_RBF9 =H_trn_c(:,9* SVD_cutoffPoint + 1: 10 * SVD_cutoffPoint);
H_RBF10 =H_trn_c(:,10* SVD_cutoffPoint + 1: 11 * SVD_cutoffPoint);

% Parameters setting (you can specify your own parameter)
a = 3.7;      % SCAD parameter
eps = 1e-3;  % error tolerance for inner loop
M_max = 1;   % maximum number of iterations for inner loop
eps_outer = 1e-3;  % error tolerance for outter loop
M_max_outer  =  10; % maximum number of iterations for outter loop
delta = 10;    %  parameter in huber loss function      
SVD_cutoffPoint =   5; % q value

% Regualrization parameters for lambda1, lambda2 and lambda3
lambda_seq = 2.^[-4:1:2]
eta_seq  = 2.^[-4:1:2]
Lambda_Gird = 2.^[-7:1:-2]


% Initilization
t = 0;
beta_sol = cell(length(lambda_seq), length(eta_seq), length(Lambda_Gird));  % Store SCAD-Ridge estimator
Temp = zeros(length(lambda_seq), length(eta_seq), length(Lambda_Gird));
NZ_sol = zeros(length(lambda_seq), length(eta_seq),length(Lambda_Gird));   % Store number of nonzeros 


alpha_SP = cell(length(lambda_seq), length(eta_seq), length(Lambda_Gird));   % Store kernel weights
IC_values = zeros(length(lambda_seq), length(eta_seq), length(Lambda_Gird)); % Store SVMIC values

start = cputime;
for alpha_ind =1 : length(Lambda_Gird)
        for s = 1 : length(lambda_seq)
                for v  = 1 : length(eta_seq)
                alpha_old = alpha_0;     
                lambda = lambda_seq(s);
                eta = eta_seq(v);
                Lambda = ones(size(alpha_old)) * Lambda_Gird(alpha_ind);
                Err_outer = 1;
                M_outer = 1;
                while Err_outer >= eps_outer && M_outer <= M_max_outer
                        H_trn_c_W = [sqrt(alpha_old(1)) * H_LK sqrt(alpha_old(2)) * H_RBF1 sqrt(alpha_old(3)) * H_RBF2 sqrt(alpha_old(4)) * H_RBF3 sqrt(alpha_old(5)) * H_RBF4... 
                        sqrt(alpha_old(6)) * H_RBF5 sqrt(alpha_old(7)) * H_RBF6 sqrt(alpha_old(8)) * H_RBF7 sqrt(alpha_old(9)) * H_RBF8 sqrt(alpha_old(10)) * H_RBF9 sqrt(alpha_old(11)) * H_RBF10];    
                        %%% Using the LASSO estiamte as the initilal estimator
                        [b fitinfo] = lasso(H_trn_c_W,y_trn_c,'CV',5);
                        lam = fitinfo.Index1SE;
                        % Scaling the Lasso estimate with blocksize
                        beta0 = b(:,lam);
                        beta = beta0; % use LASSO estiamte as the initilziation estimate   
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        Err= 1;
                        M = 1;
                        % Create Multiiple Kernel Learning with sparisty%%%%%%%%%%%%%%%%%%%%%
                        while Err >= eps && M <= M_max
                                [n_Train_H, p_train_H] = size(H_trn_c_W);
                                %%%%%%   Robust penalization using SCAD Ridge%%%%%%   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                m = abs(y_trn_c - H_trn_c * beta);
                                 % Using huber loss function
                                p1  =(( 0.5 * m.^2 .* (m <= delta) + (delta * m - delta^2/2) .*  (m >delta))) ./ (m.^2);    
                                wmatrix = diag(p1);
                                epsilon = 1e-2;
                                % SCAD-Ridge regularization
                                Ridge_SCAD = (eta * abs(beta)) .* (abs(beta)> a * lambda) + (lambda + eta * abs(beta)) .* (abs(beta) < lambda)+...
                                + (abs(beta)>= lambda & abs(beta)<= a * lambda ) .* ((a * lambda - abs(beta))/(a -1) + eta * abs(beta));                                             
                                Ridge_SCAD_vector =  Ridge_SCAD  ./ (abs(beta) + epsilon);
                                Ridge_SCAD_new = diag(Ridge_SCAD_vector);
                                %Obtain SCAD-Ridge estimator
                                beta_new = (H_trn_c_W' *  wmatrix * H_trn_c_W + 0.5* n_Train_H * Ridge_SCAD_new) \  H_trn_c_W' * wmatrix * y_trn_c;
                                Err  = sum((beta_new - beta).^2);
                                M = M + 1;
                                beta = beta_new;     
                        end 
                             Res =  wmatrix^1/2 * y_trn_c -  wmatrix^1/2 * H_trn_c_W * beta_new;     % Calculate residual
                             K = norm(wmatrix^1/2 * H_trn_c_W,2);       % Calculate step size                        
                            gradient = [ (wmatrix^1/2 *H_LK * beta_new(1:SVD_cutoffPoint))' * Res; (wmatrix^1/2 * H_RBF1 * beta_new(SVD_cutoffPoint+1:SVD_cutoffPoint*2))' * Res;
                            (wmatrix^1/2 * H_RBF2 * beta_new(SVD_cutoffPoint*2+1:SVD_cutoffPoint*3))' * Res;(wmatrix^1/2 * H_RBF3 * beta_new(SVD_cutoffPoint*3+1:SVD_cutoffPoint*4))' * Res;
                            (wmatrix^1/2 * H_RBF4 * beta_new(SVD_cutoffPoint*4+1:SVD_cutoffPoint*5))' * Res;(wmatrix^1/2 * H_RBF5 * beta_new(SVD_cutoffPoint*5+1:SVD_cutoffPoint*6))' * Res;
                            (wmatrix^1/2 * H_RBF6 * beta_new(SVD_cutoffPoint*6+1:SVD_cutoffPoint*7))' * Res; (wmatrix^1/2 * H_RBF7 * beta_new(SVD_cutoffPoint*7+1:SVD_cutoffPoint*8))' * Res;
                            (wmatrix^1/2 * H_RBF8 * beta_new(SVD_cutoffPoint*8+1:SVD_cutoffPoint*9))' * Res;(wmatrix^1/2 * H_RBF9 * beta_new(SVD_cutoffPoint*9+1:SVD_cutoffPoint*10))' * Res;
                            (wmatrix^1/2 * H_RBF10 * beta_new(SVD_cutoffPoint*10+1:SVD_cutoffPoint*11))' * Res];   % Calculate gradient
                           alpha = Func_Thresholding(alpha_old + gradient ./ (2 * sqrt(alpha_old + 1e-3) * K), 'soft_nonnegative', Lambda);   % Update of kernel weights
                            Err_outer =  sum((alpha - alpha_old).^2); 
                            M_outer = M_outer + 1;
                            alpha_old = alpha;
                end
                         % Collecting results
                         beta(abs(beta)<=1e-2) = 0;
                        beta_sol{s,v,alpha_ind} = beta;
                        alpha_SP{s,v,alpha_ind}= alpha;
                        H2 = [sqrt(alpha_old(1)) * H_LK sqrt(alpha_old(2)) * H_RBF1 sqrt(alpha_old(3)) * H_RBF2 sqrt(alpha_old(4)) * H_RBF3... 
                            sqrt(alpha_old(5)) * H_RBF4 sqrt(alpha_old(6)) * H_RBF5 sqrt(alpha_old(7)) * H_RBF6 sqrt(alpha_old(8)) * H_RBF7... 
                            sqrt(alpha_old(9)) * H_RBF8 sqrt(alpha_old(10)) * H_RBF9 sqrt(alpha_old(11)) * H_RBF10] ;
                        IC_values(s,v,alpha_ind) = length(trnInds ) * log(norm(wmatrix^1/2 * y_trn_c -  wmatrix^1/2 * H2 * beta,2)^2 /  length(trnInds))  +  sqrt(log(length(trnInds))) * NZ_sol(s,v) * log(length(trnInds)) / length(trnInds);   
                        disp(['splitTimeInd =',num2str(splitTimeInd ),'---The--',num2str(s),'--(',num2str(length(lambda_seq)),')---' ,num2str(v),'--(',num2str(length(eta_seq)),')---',num2str(alpha_ind),...
                            '--(',num2str(length(Lambda_Gird)),')---''-th itertaton has finished!'])
                 end
         end
end

%Find the optimal kernel weights
for  j = 1 : length(Lambda_Gird)
       Temp =  IC_values(:,:,j);
       betaInd_path(j) = min(min(Temp));
end
Opt_alpha =  find(betaInd_path == min(betaInd_path), 1,'first');
Temp2 = IC_values(:,:,Opt_alpha);
[Opt_i, Opt_j] = find(Temp2 == min(min(Temp2)), 1,'first');

alphaOpt =  alpha_SP{Opt_i,Opt_j,Opt_alpha};
beta_Opt = beta_sol{Opt_i,Opt_j,Opt_alpha};

alphaOptNormal = alphaOpt;

% Get H matrix for test part
H_LK = H_tst_c(:,1:SVD_cutoffPoint);
H_RBF1 = H_tst_c(:,SVD_cutoffPoint + 1: 2 * SVD_cutoffPoint);
H_RBF2 = H_tst_c(:,2 * SVD_cutoffPoint + 1: 3 * SVD_cutoffPoint);
H_RBF3 = H_tst_c(:,3 * SVD_cutoffPoint + 1: 4 * SVD_cutoffPoint);
H_RBF4 = H_tst_c(:,4 * SVD_cutoffPoint + 1: 5 * SVD_cutoffPoint);
H_RBF5 =H_tst_c(:,5 * SVD_cutoffPoint + 1: 6 * SVD_cutoffPoint);
H_RBF6 = H_tst_c(:,6 * SVD_cutoffPoint + 1: 7 * SVD_cutoffPoint);
H_RBF7 =H_tst_c(:, 7 * SVD_cutoffPoint + 1: 8 * SVD_cutoffPoint);
H_RBF8 =H_tst_c(:,8 * SVD_cutoffPoint + 1: 9 * SVD_cutoffPoint);
H_RBF9 =H_tst_c(:,9* SVD_cutoffPoint + 1: 10 * SVD_cutoffPoint);
H_RBF10 =H_tst_c(:,10* SVD_cutoffPoint + 1: 11 * SVD_cutoffPoint);

   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%        Test   Procedure     %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
H_tst_c2  = [sqrt(alphaOptNormal(1)) * H_LK sqrt(alphaOptNormal(2)) * H_RBF1 sqrt(alphaOptNormal(3)) * H_RBF2... 
            sqrt(alphaOptNormal(4)) * H_RBF3 sqrt(alphaOptNormal(5)) * H_RBF4 sqrt(alphaOptNormal(6)) * H_RBF5 sqrt(alphaOptNormal(7)) * H_RBF6... 
            sqrt(alphaOptNormal(8)) * H_RBF7 sqrt(alphaOptNormal(9)) * H_RBF8 sqrt(alphaOptNormal(10)) * H_RBF9  sqrt(alphaOptNormal(11)) * H_RBF10];

alphaOpt_SP(:,splitTimeInd) = alphaOpt;
NZ_beta(splitTimeInd) =  sum(abs(beta_Opt)>=1e-6);
NZ_alpha(splitTimeInd) =   sum(abs(alphaOpt) >=1e-6);      
         
 y_pred = H_tst_c2 * beta_Opt;
 y_forecast = y_pred+mean(y_trn);
 y_actual =   y_tst ;
 
y_forecast2{splitTimeInd} = y_pred+mean(y_trn);
y_actual2{splitTimeInd} =   y_tst ;

RMSE(splitTimeInd)=sqrt(mean( (y_forecast - y_actual).^2));
end

finish  = cputime;
used_time = (finish - start)/(nCV_outer * length(lambda_seq) * length(eta_seq) * length(Lambda_Gird));

RMSE2 = median(RMSE); NZ_beta2 = median(NZ_beta); NZ_alpha2 = median(NZ_alpha);

disp('############################################################################################################################################');
disp(['dataName = ',num2str(dataName), '---NumFeatureInSubset=', num2str(NumFeatureInSubset),'-----RMSE2  = ',num2str(RMSE2), '---Delta= ',num2str(delta),'---MaxN=',num2str(M_max),...
'---SVD_cutoffPoint=',num2str(SVD_cutoffPoint), '---NZ_beta2=',num2str(NZ_beta2),'---NZ_alpha2=',num2str(NZ_alpha2)]);
 disp('############################################################################################################################################');


 


     

