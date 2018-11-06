function [trK, tstK] = cmpExpX2Kernel(trD, tstD, gamma)    
    nTr=size(trD,2);
    nTst=size(tstD,2);

    trK=zeros(nTr,nTr);
    for i=1:nTr
        for j=(i+1):nTr
            trK(i,j)=sum((trD(:,i)-trD(:,j)).^2./(trD(:,i)+trD(:,j)+eps));
        end
    end
    if ~exist('gamma','var')
        gamma=mean(trK(trK~=0));
    end
    trK=trK+trK';
    trK=exp(-trK/gamma);

    tstK=zeros(nTr,nTst);
    for i=1:nTr
        for j=1:nTst
            tstK(i,j)=sum((trD(:,i)-tstD(:,j)).^2./(trD(:,i)+tstD(:,j)+eps));
        end
    end
    tstK=exp(-tstK/gamma);
end