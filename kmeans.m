function [C, A, iter]=kmeans(X, k, maxIter, centroids)
    [d,n]=size(X);

    if ~exist('centroids','var')
        centroids=randperm(n,k);
    end
    C=X(:,centroids);

    A=nan(n,1);

    for iter=1:maxIter
        D=pdist2(X',C','euclidean');
        [~,Anew]=min(D,[],2);
        change=sum(Anew~=A);
        A=Anew;
        if change==0
            break;
        end

        for i=1:k
            idx=find(A==i);
            C(:,i)=mean(X(:,idx),2);        
        end
    end
end