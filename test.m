            nTrain=size(trD,2);
            nTest=size(tstD,2);

            %% 5 Fold split
            nFold=5;
            IDXes=cell(nFold,1);
            for i=1:8
                idx_i=find(trLbs==i);
                nSample=length(idx_i);
                order=1:nSample;
                for j=1:nFold
                   IDXes{j}=[IDXes{j};idx_i(order(((j-1)*ceil(nSample/nFold)+1):min(end,j*ceil(nSample/nFold))))];
                end
            end

            %% RBF kernel
            pred={};acc={};dec={};
            for i=1:nFold
                tridx=unique(cat(1,IDXes{setdiff(1:nFold,i)}));
                vlidx=unique(cat(1,IDXes{i}));
                model = svmtrain(trLbs(tridx), trD(:,tridx)', '-s 0 -t 2 -q');
                [pred{i}, acc{i}, dec{i}] = svmpredict(trLbs(vlidx), trD(:,vlidx)', model);
            end
            cv_acc(iC,iG)=sum(cellfun(@(x)x(1),acc).*cellfun(@(x)length(x),pred))/nTrain;
            disp(cv_acc);
            
            
            pred={};acc={};dec={};
            for i=1:nFold
                tridx=unique(cat(1,IDXes{setdiff(1:nFold,i)}));
                vlidx=unique(cat(1,IDXes{i}));
                model = svmtrain(trLbs(tridx), trD(:,tridx)', '-s 0 -t 2 -g 10 -c 1000 -q');
                [pred{i}, acc{i}, dec{i}] = svmpredict(trLbs(vlidx), trD(:,vlidx)', model);
            end
            cv_acc=sum(cellfun(@(x)x(1),acc).*cellfun(@(x)length(x),pred))/nTrain;
            disp(cv_acc);


            %% exponential X2 kernel
            [trK, tstK] = cmpExpX2Kernel(trD, tstD, 5);
            pred={};acc={};dec={};
            for i=1:nFold
                tridx=unique(cat(1,IDXes{setdiff(1:nFold,i)}));
                vlidx=unique(cat(1,IDXes{i}));
                model = svmtrain(trLbs(tridx), [(1:length(tridx))',trK(tridx,tridx)'], '-q -s 0 -t 4 -c 1000');
                [pred{i}, acc{i}, dec{i}] = svmpredict(trLbs(vlidx), [(1:length(vlidx))',trK(tridx,vlidx)'], model);
            end
            cv_acc=sum(cellfun(@(x)x(1),acc).*cellfun(@(x)length(x),pred))/nTrain;

            disp(cv_acc);


            %% test score
            model = svmtrain(trLbs, [(1:nTrain)',trK'], '-q -s 0 -t 4 -c 1000');
            [pred, acc, dec] = svmpredict(ones(1,nTest)', [(1:nTest)',tstK'], model);
            %% write predicition into a csv file
            csvwrite('predTestLabels.csv',[tstIds',pred]);