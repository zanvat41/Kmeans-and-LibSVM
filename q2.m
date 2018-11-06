X = load('../digit/digit.txt');
X = X';
Y = load('../digit/labels.txt');
[d,n]=size(X);

for k=[2,4,6]
    maxIter=20;
    centroids=1:k;
    [C, A, iter]=kmeans(X, k, maxIter, centroids);
    
    SS=nan(1,k);
    for i=1:k
        inx=find(A==i);
        SS(i)=sum(pdist2(X(:,inx)',C(:,i)','euclidean').^2);
    end
    totalSS=sum(SS);
   
    sameClass=(pdist2(Y,Y,@(x,y) x-y))==0;
    sameClu=(pdist2(A,A,@(x,y) x-y))==0;
    sameClassSameClu = sum(sum(sameClass & sameClu))-n;
    sameClassDiffClu = sum(sum(sameClass & ~sameClu));
    diffClassSameClu = sum(sum(~sameClass & sameClu))-n;
    diffClassDiffClu = sum(sum(~sameClass & ~sameClu));
    p1=sameClassSameClu/(sameClassSameClu+sameClassDiffClu);
    p2=diffClassDiffClu/(diffClassDiffClu+diffClassSameClu);
    
    p3=(p1+p2)/2;
    
    fprintf('\nk = %d:\n',k);
    fprintf('iter = %d;\n',iter);
    fprintf('totalSS = %.2f;\n',totalSS);
    fprintf('p1 = %.2f%%, p2 = %.2f%%, p3 = %.2f%%.\n\n',p1*100, p2*100, p3*100);    
end

totalSS=[];
p1=[];p2=[];
for k=1:10
    for time=1:10
        maxIter=20;
        [C, A, iter]=kmeans(X, k, maxIter);
        
        SS=nan(1,k);
        for i=1:k
            inx=find(A==i);
            SS(i)=sum(pdist2(X(:,inx)',C(:,i)','euclidean').^2);
        end
        totalSS(k,time)=sum(SS);

        sameClass=(pdist2(Y,Y,@(x,y) x-y))==0;
        sameClu=(pdist2(A,A,@(x,y) x-y))==0;
        sameClassSameClu = sum(sum(sameClass & sameClu))-n;
        sameClassDiffClu = sum(sum(sameClass & ~sameClu));
        diffClassSameClu = sum(sum(~sameClass & sameClu))-n;
        diffClassDiffClu = sum(sum(~sameClass & ~sameClu));
        p1(k,time)=sameClassSameClu/(sameClassSameClu+sameClassDiffClu);
        p2(k,time)=diffClassDiffClu/(diffClassDiffClu+diffClassSameClu);
    end
end
p3=(p1+p2)/2;

figure(1);
plot(mean(totalSS,2),'*-');
xlabel('K'); legend('SS_{total}');

figure(2);
plot(squeeze(mean(cat(3,p1,p2,p3),2)),'*-')
xlabel('K'); legend('p_1','p_2','p_3');