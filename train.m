function [MAP_result] = train(XTrain, YTrain, param,LTrain,XTest,LTest,anchor,Binit,Vinit,Pinit)

%% set the parameters
nbits = param.nbits;
mu = param.mu;
beta = param.beta;
alpha = param.alpha;
delta = param.delta;
chunk = param.chunk;
gama = param.gama;
lamda = param.lamda;
paramiter = param.paramiter;
nq = param.nq;

MAP_result=zeros(1,8);  

%% get the dimensions of features
n = size(XTrain,1); 
dX = size(anchor,1); 
dY = size(YTrain,2);


%% initialization
n1 = floor((nq/chunk)*nq);
n2 = nq-n1;
nmax = 1000;
A = zeros(n,dY);
myindex = zeros(8,nmax);
SA = zeros(n,dY);
normytagA = ones(chunk,1);
normdiff = zeros(1,chunk);

B = Binit;
V = Vinit;
P = Pinit;

%% iterative optimization
for round = 1:(n/chunk)
    fprintf('chunk %d: training starts. \n',round)
    e_id = (round-1)*chunk+1;
    n_id = min(round*chunk,n);
    if round == 8 
        n_id = n;
    end
    X = Kernelize(XTrain(e_id:n_id,:),anchor);
    nsample = n_id-e_id+1;  
    
    if round == 1
        R = YTrain(e_id:n_id,:)'*YTrain(e_id:n_id,:);
        [RA,~]=mapminmax(R,0,1);
        A(e_id:n_id,:) = (lamda*YTrain(e_id:n_id,:)+YTrain(e_id:n_id,:)*RA)/lamda;
        
        diff = A(e_id:n_id,:) - YTrain(e_id:n_id,:);
        for j = 1:nsample
            normdiff(j) = norm(diff(j,:),2);
        end
        [~,index] = sort(normdiff(:,1:nsample));
        myindex(round,:) = index(:,1:nmax);
        
        for i =1:nsample
            if norm(A(i+(round-1)*chunk,:))~=0
                normytagA(i,:)=norm(A(i+(round-1)*chunk,:));
            end
        end
        normytagA = repmat(normytagA,1,dY);
        SA(e_id:n_id,:) = A(e_id:n_id,:)./normytagA;
       
        for iter = 1:paramiter
            C3 = X'*X;
            C4 = X'*B(e_id:n_id,:);
            
            % update P
            P = pinv(C3+(delta/mu)*eye(dX))*(C4);
            
            % update V
            C = alpha*nbits*(2*SA(e_id:n_id,:)*(SA(e_id:n_id,:)'*B(e_id:n_id,:)) ...
            -ones(nsample,1)*(ones(1,nsample)*B(e_id:n_id,:)))+ beta*B(e_id:n_id,:);
        
            Temp = C'*C-1/nsample*(C'*ones(nsample,1)*(ones(1,nsample)*C));
            [~,Lmd,QQ] = svd(Temp); clear Temp
            idx = (diag(Lmd)>1e-6);
            Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
            PP = (C-1/nsample*ones(nsample,1)*(ones(1,nsample)*C)) *  (Q / (sqrt(Lmd(idx,idx))));
            P_ = orth(randn(nsample,nbits-length(find(idx==1))));
            V(e_id:n_id,:) = sqrt(nsample)*[PP P_]*[Q Q_]';
            
            % update B
            B(e_id:n_id,:) = sign(alpha*nbits*(2*SA(e_id:n_id,:)*(SA(e_id:n_id,:)'*V(e_id:n_id,:))...
            -ones(nsample,1)*(ones(1,nsample)*V(e_id:n_id,:)))+...
                beta*V(e_id:n_id,:) +mu*X*P );
            
            
        end
        Qq = A(myindex(1,1:nq),:);
        Btemp = B(e_id:n_id,:);
        Bq = Btemp(myindex(1,1:nq),:);

    end
    
    if round >= 2
        
        RR = R;
        R = RR + YTrain(e_id:n_id,:)'*YTrain(e_id:n_id,:);
        [RA,~]=mapminmax(R,0,1);
        A(e_id:n_id,:) = (lamda*YTrain(e_id:n_id,:)+YTrain(e_id:n_id,:)*RA)/lamda;
        
        diff = A(e_id:n_id,:) - YTrain(e_id:n_id,:);
        normdiff = zeros(1,chunk);
        for j = 1:nsample
            normdiff(j) = norm(diff(j,:),2);
        end
        [~,index] = sort(normdiff(:,1:nsample));
        myindex(round,:) = index(:,1:nmax);
        
        normytagA = ones(chunk,1);
        for i =1:nsample
            if norm(A(i+(round-1)*chunk,:))~=0
                normytagA(i,:)=norm(A(i+(round-1)*chunk,:));
            end
        end
        normytagA = repmat(normytagA,1,dY);
        SA(e_id:n_id,:) = A(e_id:n_id,:)./normytagA;     
        
        CC3 = C3; CC4 = C4; 
        for iter = 1:paramiter
            
            C3 = CC3+X'*X;
            C4 = CC4+X'*B(e_id:n_id,:);
            
            % update P
            P = pinv(C3+(delta/mu)*eye(dX))*(C4);   
            
            % update V
            C = alpha*nbits*(2*SA(e_id:n_id,:)*(SA(e_id:n_id,:)'*B(e_id:n_id,:)) ...
            -ones(nsample,1)*(ones(1,nsample)*B(e_id:n_id,:))) + beta*B(e_id:n_id,:) ...
            + gama*nbits*(2*SA(e_id:n_id,:)*(Qq'*Bq)...
            -ones(nsample,1)*(ones(1,nq)*Bq));
  
            Temp = C'*C-1/nsample*(C'*ones(nsample,1)*(ones(1,nsample)*C));
            [~,Lmd,QQ] = svd(Temp); clear Temp
            idx = (diag(Lmd)>1e-6);
            Q = QQ(:,idx); Q_ = orth(QQ(:,~idx));
            PP = (C-1/nsample*ones(nsample,1)*(ones(1,nsample)*C)) *  (Q / (sqrt(Lmd(idx,idx))));
            P_ = orth(randn(nsample,nbits-length(find(idx==1))));
            V(e_id:n_id,:) = sqrt(nsample)*[PP P_]*[Q Q_]';
             
            % update B
            B(e_id:n_id,:) = sign(alpha*nbits* (2*SA(e_id:n_id,:)*(SA(e_id:n_id,:)'*V(e_id:n_id,:))...
            -ones(nsample,1)*(ones(1,nsample)*V(e_id:n_id,:))) + ...
                beta*V(e_id:n_id,:)   +mu*X*P);
            
            
        end
        % update Qq
        yindex = myindex(round,1:n2) + (round-1)*chunk;
        neighbor = A(yindex,:);
        olddata = Qq(randsample(nq,n1),:);
        Qq = [olddata;neighbor];
         
        % update Bq
        oldBq = Bq(randsample(nq,n1),:);  
        Btemp = B(e_id:n_id,:);
        Bq = [oldBq;Btemp(myindex(round,1:n2),:)]; 
       
    end
    
    fprintf('       : training ends, evaluation begins. \n')
    XKTest=Kernelize(XTest,anchor);
    BxTest = compactbit(XKTest*P >= 0);
    BxTrain = compactbit(B(1:n_id,:) >= 0);
    DHamm = hammingDist(BxTest, BxTrain);
    [~, orderH] = sort(DHamm, 2);
    MAP  = mAP(orderH', LTrain(1:n_id,:), LTest);
    fprintf('       : evaluation ends, MAP is %f\n',MAP);
    MAP_result(1,round)=MAP;
    

end
