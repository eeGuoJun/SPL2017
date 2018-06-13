%% script for testing code of CADL

clear all; 
clc;
warning off

%% load toolkits
addpath('.\large_scale_svm');
addpath('.\otherFiles');

%% load dataset
addpath('.\data');

for dataset=[2] % AR Dataset
    
    if (dataset == 1)       
       load ExYaleBdata;
       num_atom_per_class = 10;
       dname = 'Extended Yale B dataset';
    elseif (dataset == 2)
        load ARdata;
        num_atom_per_class = 5;
        dname = 'AR dataset';
    elseif (dataset ==3)
         load Caldata;
         num_atom_per_class = 10;
         dname = 'Cal dataset';
    elseif (dataset ==4)
         load SCENEdata;
         num_atom_per_class = 10;
         dname = 'SCENE dataset';
    elseif (dataset ==5)
         load UCFdata;
         num_atom_per_class = 10;
         dname = 'UCF dataset';
    elseif (dataset == 5)
         load session1_05_1_netural_14
         Train_DAT = double(DAT(:,labels<=60));
         trainlabels = labels(:,labels<=60);
         clear DAT labels;
         load session3_05_1_netural_10o 
         Test_DAT = double(DAT(:,labels<=60));
         testlabels = labels(:,labels<=60);
         clear DAT labels;    
         num_atom_per_class = 14;
         dname = 'Multi-PIE dataset';
    end
    
    tr_dat = Train_DAT;
    tt_dat = Test_DAT;
    trls = trainlabels;
    ttls = testlabels;    
    clear Train_DAT Test_DAT trainlabels testlabels;    
% %====== reduce the dimension if needed ==========
% rdim = 300 or 1000 for different datasets
% Vt = Eigenface_f(tr_dat,rdim);
% tr_dat = Vt'*tr_dat;
% tt_dat = Vt'*tt_dat;
% %================================================    
    tr_dat = normcol_equal(tr_dat);
    tt_dat = normcol_equal(tt_dat);
    
    %% set parameters
    lambda1    =   5e-3;
    lambda2    =   5e-4;
    lambda3    =   5e-4;
    max_iter   =   10;
    theta      =   5;     
    fprintf('\nlambda1 = %d, lambda2 = %d, lambda3 = %d, max_iter = %d\n',lambda1,lambda2,lambda3,max_iter);
    
    %% initialize sub-dictionaries via PCA
    fprintf('\n------------------------Initializing Dictionary------------------------\n');
    Dini = [];
    num_class   = length(unique(trls));
    num_atom_ci = num_atom_per_class;
    Dim         = size(tr_dat,1);
    DictSize    = num_class*ceil(Dim/num_class);%number of rows in analysis dictionary 
    tr_label = [];
    for ci=1:num_class
        TempData       = tr_dat(:,trls==ci);
        TempLabel      = trls(:,trls==ci);
        DataMat{ci}    = TempData;
        tr_label       = [tr_label,TempLabel];
        randn('seed',ci);                        
        DictMat{ci}    = normcol_equal(randn(ceil(Dim/num_class),Dim));
        Coef_ini{ci}   = DictMat{ci}*TempData;
        TempDataC      = tr_dat(:,trls~=ci);
        DataInvMat{ci} = inv(TempData*TempData'+lambda1*TempDataC*TempDataC'+lambda3*eye(size(TempData,1)));
    end

    %% run algorithm
    fprintf('\n\n----------------------------Algorithm CADL----------------------------');
    tic;
    [AnalyDict,Z,U,b,class_list]  = trainCADL(tr_dat,DataMat,tr_label,...
        DictMat,Coef_ini,DataInvMat,lambda2,theta,max_iter);
    temp1 = toc;
    fprintf('\nThe Trainning Time is %5.4f ', temp1)
    fprintf('\nCADL Model Training is Completed!')
    
    %% encode the testing data
    fprintf('\n\n--------------------------------Testing--------------------------------');
    tic;    
    Z_test = AnalyDict*tt_dat;
    [ttls_pred, ~] = li2nsvm_multiclass_fwd(Z_test', U, b, class_list);
    reco_rate = (sum(ttls_pred'==ttls))/length(ttls);
    fprintf(['\nThe Recognition Rate on the ', dname, ' is ', num2str(reco_rate,4)])
    temp2 = toc;
    fprintf('\nThe Testing Time is %5.4f \n', temp2)

end % for testing AR dataset