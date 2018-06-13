function [AnalyTotal,Z,U,b,class_list]=trainCADL(tr_dat,training_data,training_label,AnalyMat,Zinit,DataInvMat,lambda2,theta,maxiter)
%% Training of CADL

if nargin < 9 || isempty(maxiter)
    maxiter = 20;
end
tol = 1e-3;
tau = 1/theta; 
X = training_data;
y = training_label;
m = size(X{1},1); %dimension
n = length(y); %total number of samples
K = size(AnalyMat,2); %number of classes
DictSize = K*ceil(m/K); %number of row in analysis dictionary
class_list  = unique(y,'stable');
class_num   = length(class_list);
class_space = 1;
class_idx   = zeros(n, 1);

% define the label matrix Y_label for c two-class classification problems (one-vs-all)
Y = zeros(n, class_num);
for i = 1:n
    for j = 1 : class_space
        if y(i) == class_list(j) 
            class_idx(i) = j;
        end
    end
    if class_idx(i) == 0
        class_space = class_space + 1;
        class_idx(i) = class_space;
    end
    Y(i, class_idx(i)) = 1;
end
Y_label = sign(Y-0.5);

%% initialization
Uinit = zeros(DictSize,class_num);
binit = zeros(1,class_num);
Zk = Zinit;
Uk = Uinit;
bk = binit;
AnalyDictk = AnalyMat;
AnalyTotal = cell2mat(AnalyDictk');

Ztotal=zeros(DictSize,n);
k = 0;
rel_deltaD = 1;
while k < maxiter && rel_deltaD > tol
    k = k+1;
    fprintf('\nIteration: %i, ', k);    
    row_ID=0;
    column_ID=0;
    if k~=1      
        loss_oneclass = [];
        for i = 1 : n         
            Y_labelki = Ztotal(:,i)'*Uk + bk;
            loss_idx  = find( Y_labelki.*Y_label(i,:) < 1 ) ;% hinge loss function
            if isempty(loss_idx)
               Ztotal(:,i) = AnalyTotal*tr_dat(:,i);                           
            else                
                Yi_idx = Y_label(i,loss_idx);
                Uk_idx = Uk(:,loss_idx); 
                bk_idx = bk(loss_idx);
                mk=inv(eye(size(Uk_idx,1))+2*lambda2*theta*Uk_idx*Uk_idx');
                Ztotal(:,i)=mk*(AnalyTotal*tr_dat(:,i)-2*lambda2*theta*Uk_idx*bk_idx'+2*lambda2*theta*Uk_idx*Yi_idx');
                loss_oneclass = [loss_oneclass (Yi_idx*(Uk_idx'*Ztotal(:,i)+bk_idx')-1)^2];       
            end
        end
    else
        Ztotal = AnalyTotal*tr_dat;
    end
   
    for ci =1:K
        row_per_class = size(Zk{ci},1);
        column_per_class = size(Zk{ci},2);     
        if k~=1
           Zk{ci}= Ztotal(row_ID+1:row_ID+row_per_class,column_ID+1:column_ID+column_per_class);
        end
       %% update Dk
        AnalyDictkm1{ci}=AnalyDictk{ci}; %row-wise normalization
        AnalyDictk{ci}=Zk{ci}*X{ci}'*DataInvMat{ci};     
        norm_AnalyDictk{ci} = normcol_equal(AnalyDictk{ci}');%
        AnalyDictk{ci}= norm_AnalyDictk{ci}';%omege_k, each row's l2-norm = 1   
        AnalyTotal=cell2mat(AnalyDictk');
        rel_deltaD = norm(AnalyDictk{ci}(:)-AnalyDictkm1{ci}(:))/norm(AnalyDictk{ci}(:));

        row_ID =row_ID + row_per_class;
        column_ID = column_ID + column_per_class;   

    end

    %% update Uk bk
    [Uk, bk, ~] = li2nsvm_multiclass_lbfgs(Ztotal',y, tau);
    
end
AnalyDict = AnalyDictk;
Z = Ztotal;
U = Uk;
b = bk;
fprintf('Done!');

end