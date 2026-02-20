% w is a specific dataset 
% ensure reproducability 
rng(42,'twister')
close all
clc;
set(0,'defaultTextInterpreter','latex')

% read in data and remove non-cultural features + features missing > 50% of
% possible values

% from EA
KinOrgRaw = readtable('~/Documents/KinshipOrgDataRaw.csv');
SubRaw = readtable('~/Documents/SubsistenceDataRaw.csv');
% EA_All_Raw = readtable('~/Documents/EA_All.csv');
% From Pulotu
IsoRaw = readtable('~/Documents/IsoDataRaw.csv');
ReligRaw = readtable('~/Documents/RelDataRaw.csv');
Binford = readtable('~/Documents/Binford_all.csv');
Seshat = readtable('~/Documents/Seshat.csv');
Birds = readtable('~/Documents/data_birds_OH.csv');

load('~/Documents/data.mat');
Genetics = data;
% EA_All = OHspecial_transform(EA_All_Raw);
KinOrg = OHspecial_transform(KinOrgRaw);
Sub = OHspecial_transform(SubRaw);
Iso = OHspecial_transform(IsoRaw);
Relig = OHspecial_transform(ReligRaw);


for w = 2:2
  
    if w == 1
        X = KinOrg;
    elseif w == 2
        X = Sub;
    elseif w == 3
        X = Relig;
    elseif w == 4
        X = Iso;
    elseif w == 5
        X = Binford;
    elseif w == 6
        X = Seshat;
    elseif w == 7
        X = Birds;
    elseif w == 8
        X = Genetics;
    end
    
% remove features missing > 50% of their entries 
    if w ~= 8
        dim = size(X);
        X = X(:,1:dim(2));
        cols = X.Properties.VariableNames;
        X = table2array(X);
        for j = 1:dim(2)-1
            temp = double(isnan(X(:,j)));
            idx = find(temp == 1);
            counts(j) = length(idx);
        end
        counts = counts./dim(1);
        idx = find(counts > .5);
        cols(idx) = [];
        X(:,idx) = [];
    end


    if w ~= 8
        dim = size(X);
    elseif w == 8
        dim = size(X');
    end
   
    if w ~= 8
    MnInit = repmat(nanmean(X),dim(1),1);
    STDInit = repmat(nanstd(X),dim(1),1);
    end
   if w < 5
       X_ = (X-MnInit);
   elseif w == 5
       X_ = X;
   elseif w == 6
       X_ = (X - MnInit)./STDInit;
   elseif w == 7
       X_ = X - MnInit;
   elseif w == 8
       X_ = X;
   end

    if w ~= 8
    X_ = X_';
    end
    
    MaxStop = min(dim);
    if MaxStop > 75
        MaxStop = 75;
    end
    zz = 2:MaxStop;
    rms = zeros([1,MaxStop-1]);
    for y = 1:MaxStop-1
       

        z = zz(y);
        fprintf('###############################\n')
        fprintf('number of PCs: %i\n',z)
        fprintf('###############################\n')
        if y > 1
            pause(1)
        end
        for k = 1:1 % change if you want to do replications 

            opts = struct( 'maxiters', 80,...
               'algorithm', 'vb',...
               'uniquesv', 0,...
               'rmsstop',[ 80 eps eps ],...
               'cfstop',[80 0 0],...
               'minangle', 0 );
            [ A, S, Mu, V, cv, hp, lc ] = pca_full( X_, z, opts );
            if z == 20
                figure
                scatter(S(1,:),S(2,:))
            end
            fprintf('Learning is finished.\n')

 
            Xrec = repmat(Mu,1,dim(1)) + A*S;
           
            if w ~= 8
            accu_num = abs(round(Xrec'+MnInit)-(X));
            accu_num = length(find(accu_num > 0));
            accu(y) = 1-accu_num./length(find(isnan(X_(:)) == 0))
            end
            rms(y) = lc.rms(end);
            rmsA(y,:) = lc.rms;
            costA(y,:) = lc.cost;
            % vv = zeros(1,MaxStop);
            % v = eig(cov(Xrec'));
            % v = flipud(v);
            % v = real(v);
            % v = v(1:MaxStop);
            % v = round(v,4);
            % v = v/sum(v);
            % vv(1:length(v)) = v;
            % var_exp(y,:) = vv; 

        end
        if y > 1 && diff(rms(y-1:y)) > 0
            break
        end


    end

end
    rms = rms(rms>0);
        if w== 1
        [ AKin, SKin, MuKin, VKin, cvKin, hpKin, lcKin ] = pca_full( X_, zz(y-1), opts );
        Xrec = repmat(MuKin,1,dim(1)) + AKin*SKin;
        for i = 1:size(X_,1)
            for j = 1:size(X_,2)
                    Vr(i,j) = AKin(i,:) * cvKin.S{j} * AKin(i,:)' + ...
                        SKin(:,j)' * cvKin.A{i} * SKin(:,j) + ...
                    sum( sum( cvKin.S{j} .* cvKin.A{i} ) ) + cvKin.Mu(i);
            end
        end
        VrKin = Vr;
         T = array2table(Xrec');
        T.Properties.VariableNames = cols;
        writetable(T,'EA_VBPCA_Recon_Kinship_Org.csv')
        XrecKin = Xrec;
        rmsKin = rms;
        rmsAKin = rmsA;
        costAKin = costA;
        % var_exp_Kin = var_exp;
        accu_Kin = accu;
        figure
        scatter(SKin(1,:),SKin(2,:),'filled')
        title('Kinship')
        X_uncent_kin = round(XrecKin'+MnInit);
        for i = 1:size(X_uncent_kin,1)
            for j = 1:size(X_uncent_kin,2)
                if isnan(X(i,j)) == 0
                    X_uncent_kin(i,j) = X(i,j);
                end
            end
        end
         TT = array2table(X_uncent_kin);
        TT.Properties.VariableNames = cols;
        writetable(TT,'EA_VBPCA_Kin_binary.csv')
        
        figure
        hold on
        box on
        grid on
        yyaxis left
        plot(2:length(accu_Kin)+1,accu_Kin,'linewidth',2,'Marker','o')
        ylabel('Accuracy')
        yyaxis right
        plot(2:length(accu_Kin)+1,rmsKin,'linewidth',2,'Marker','o')
        xline(length(accu_Kin),'linewidth',2,'Color','black')
        xlabel('Number of PCs')
        ylabel('RMS')
        xlim([2,length(accu_Kin)+1])
        hold off
        title('Kinship, Inheritence \& Community Organization')
        set(gca,'FontSize', 18)

        clear rmsA
        clear costA
        clear var_exp
        clear Vr
        clear accu
        elseif w ==2
        [ ASub, SSub, MuSub, VSub, cvSub, hpSub, lcSub ] = pca_full( X_, zz(y-1), opts );
         Xrec = repmat(MuSub,1,dim(1)) + ASub*SSub;
         for i = 1:size(X_,1)
            for j = 1:size(X_,2)
                    Vr(i,j) = ASub(i,:) * cvSub.S{j} * ASub(i,:)' + ...
                        SSub(:,j)' * cvSub.A{i} * SSub(:,j) + ...
                    sum( sum( cvSub.S{j} .* cvSub.A{i} ) ) + cvSub.Mu(i);
            end
         end


         T = array2table(Xrec');
        T.Properties.VariableNames = cols;
        VrSub = Vr;
        writetable(T,'EA_VBPCA_Recon_Subsist.csv')
        XrecSub = Xrec;
        rmsSub = rms;
        rmsASub = rmsA;
        costASub = costA;
        % var_exp_Sub = var_exp;
        accu_Sub = accu;
        figure
        scatter(SSub(1,:),SSub(2,:),'filled')
        title('Subsistence')
         X_uncent_sub = round(XrecSub'+MnInit);
        for i = 1:size(X_uncent_sub,1)
            for j = 1:size(X_uncent_sub,2)
                if isnan(X(i,j)) == 0
                    X_uncent_sub(i,j) = X(i,j);
                end
            end
        end
                 TT = array2table(X_uncent_sub);
        TT.Properties.VariableNames = cols;
        writetable(TT,'EA_VBPCA_Sub_binary.csv')

        figure
        hold on
        box on
        grid on
        yyaxis left
        plot(2:length(accu_Sub)+1,accu_Sub,'linewidth',2,'Marker','o')
        ylabel('Accuracy')
        yyaxis right
        plot(2:length(accu_Sub)+1,rmsSub,'linewidth',2,'Marker','o')
        xline(length(accu_Sub),'linewidth',2,'Color','black')
        xlabel('Number of PCs')
        ylabel('RMS')
        xlim([2,length(accu_Sub)+1])
        hold off
        title('Subsistence \& Labor')
        set(gca,'FontSize', 18)

        clear rmsA
        clear costA
        clear var_exp
        clear Vr
        clear accu
      
        elseif w == 3
              [ ARel, SRel, MuRel, VRel, cvRel, hpRel, lcRel ] = pca_full( X_, zz(y-1), opts );
         Xrec = repmat(MuRel,1,dim(1)) + ARel*SRel;
        for i = 1:size(Xrec,1)
            for j = 1:size(Xrec,2)
                    Vr(i,j) = ARel(i,:) * cvRel.S{j} * ARel(i,:)' + ...
                        SRel(:,j)' * cvRel.A{i} * SRel(:,j) + ...
                    sum( sum( cvRel.S{j} .* cvRel.A{i} ) ) + cvRel.Mu(i);
            end
        end
        VrRel = Vr;
        T = array2table(Xrec');
        T.Properties.VariableNames = cols;
        writetable(T,'PUL_VBPCA_Recon_Rel.csv')
        XrecRel = Xrec;
        rmsRel = rms;
        rmsARel = rmsA;
        costARel = costA;
        var_exp_Rel = var_exp;
        accu_Rel = accu;
        figure
        scatter(SRel(1,:),SRel(2,:),'filled')
        title('Religion')
         X_uncent_rel = round(XrecRel'+MnInit);
        for i = 1:size(X_uncent_rel,1)
            for j = 1:size(X_uncent_rel,2)
                if isnan(X(i,j)) == 0
                    X_uncent_rel(i,j) = X(i,j);
                end
            end
        end
             TT = array2table(X_uncent_rel);
        TT.Properties.VariableNames = cols;
        writetable(TT,'EA_VBPCA_Rel_binary.csv')

        figure
        hold on
        box on
        grid on
        yyaxis left
        plot(2:length(accu_Rel)+1,accu_Rel,'linewidth',2,'Marker','o')
        ylabel('Accuracy')
        yyaxis right
        plot(2:length(accu_Rel)+1,rmsRel,'linewidth',2,'Marker','o')
        xline(length(accu_Rel),'linewidth',2,'Color','black')
        xlabel('Number of PCs')
        ylabel('RMS')
        xlim([2,length(accu_Rel)+1])
        hold off
        title('Belief \& Practice')
        set(gca,'FontSize', 18)

        clear rmsA
        clear costA
        clear var_exp 
        clear Vr 
        clear accu
        elseif w == 4
            [ AIso, SIso, MuIso, VIso, cvIso, hpIso, lcIso ] = pca_full( X_, zz(y-1), opts );
         Xrec = repmat(MuIso,1,dim(1)) + AIso*SIso;
        for i = 1:size(Xrec,1)
            for j = 1:size(Xrec,2)
                    Vr(i,j) = AIso(i,:) * cvIso.S{j} * AIso(i,:)' + ...
                        SIso(:,j)' * cvIso.A{i} * SIso(:,j) + ...
                    sum( sum( cvIso.S{j} .* cvIso.A{i} ) ) + cvIso.Mu(i);
            end
        end
        VrIso = Vr;
        T = array2table(Xrec');
        T.Properties.VariableNames = cols;
        writetable(T,'PUL_VBPCA_Recon_Iso.csv')
        XrecIso = Xrec;
        rmsIso = rms;
        rmsAIso = rmsA;
        costAIso = costA;
        % var_exp_Iso = var_exp;
        accu_Iso = accu;
        figure
        scatter(SIso(1,:),SIso(2,:),'filled')
        title('Isolation')
        X_uncent_iso = round(XrecIso'+MnInit);
        for i = 1:size(X_uncent_iso,1)
            for j = 1:size(X_uncent_iso,2)
                if isnan(X(i,j)) == 0
                    X_uncent_iso(i,j) = X(i,j);
                end
            end
        end
                 TT = array2table(X_uncent_iso);
        TT.Properties.VariableNames = cols;
        writetable(TT,'EA_VBPCA_Iso_binary.csv')

        figure
        hold on
        box on
        grid on
        yyaxis left
        plot(2:length(accu_Iso)+1,accu_Iso,'linewidth',2,'Marker','o')
        ylabel('Accuracy')
        yyaxis right
        plot(2:length(accu_Iso)+1,rmsIso,'linewidth',2,'Marker','o')
        xline(length(accu_Iso),'linewidth',2,'Color','black')
        xlabel('Number of PCs')
        ylabel('RMS')
        xlim([2,length(accu_Iso)+1])
        hold off
        title('Conflict \& Contact')
        set(gca,'FontSize', 18)

        clear rmsA
        clear costA
        clear var_exp 
        clear Vr 
        clear accu
        
        elseif w == 5
         [ AEA, SEA, MuEA, VEA, cvEA, hpEA, lcEA ] = pca_full( X_, zz(y-1), opts );
         Xrec = repmat(MuEA,1,dim(1)) + AEA*SEA;
        for i = 1:size(Xrec,1)
            for j = 1:size(Xrec,2)
                    Vr(i,j) = AEA(i,:) * cvEA.S{j} * AEA(i,:)' + ...
                        SEA(:,j)' * cvEA.A{i} * SEA(:,j) + ...
                    sum( sum( cvEA.S{j} .* cvEA.A{i} ) ) + cvEA.Mu(i);
            end
        end
        VrEA = Vr;
        T = array2table(Xrec');
        T.Properties.VariableNames = cols;
        writetable(T,'EA_VBPCA_Recon_All.csv')
        XrecEA = Xrec;
        rmsEA = rms;
        rmsAEA = rmsA;
        costAEA = costA;
        var_exp_EA = var_exp;
        accu_EA = accu;
        figure
        scatter(SEA(1,:),SEA(2,:),'filled')
        title('Isolation')
        X_uncent_EA = round(XrecEA'+MnInit);
        for i = 1:size(X_uncent_EA,1)
            for j = 1:size(X_uncent_EA,2)
                if isnan(X(i,j)) == 0
                    X_uncent_EA(i,j) = X(i,j);
                end
            end
        end
                 TT = array2table(X_uncent_EA);
        TT.Properties.VariableNames = cols;
        writetable(TT,'EA_VBPCA_All_binary.csv')

        figure
        hold on
        box on
        grid on
        yyaxis left
        plot(2:length(accu_EA)+1,accu_EA,'linewidth',2,'Marker','o')
        ylabel('Accuracy')
        yyaxis right
        plot(2:length(accu_EA)+1,rmsEA,'linewidth',2,'Marker','o')
        xline(length(accu_EA),'linewidth',2,'Color','black')
        xlabel('Number of PCs')
        ylabel('RMS')
        xlim([2,length(accu_EA)+1])
        hold off
        title('EA')
        set(gca,'FontSize', 18)
        
        clear rmsA
        clear costA
        clear var_exp 
        clear Vr 
        clear accu

         elseif w == 6
         [ ASes, SSes, MuSes, VSes, cvSes, hpSes, lcSes] = pca_full( X_, zz(y-1), opts );
         Xrec = repmat(MuSes,1,dim(1)) + ASes*SSes;
        for i = 1:size(Xrec,1)
            for j = 1:size(Xrec,2)
                    Vr(i,j) = ASes(i,:) * cvSes.S{j} * ASes(i,:)' + ...
                        SSes(:,j)' * cvSes.A{i} * SSes(:,j) + ...
                    sum( sum( cvSes.S{j} .* cvSes.A{i} ) ) + cvSes.Mu(i);
            end
        end
        VrSes = Vr;
        T = array2table(Xrec');
        T.Properties.VariableNames = cols;
        writetable(T,'Seshat_VBPCA_Recon_All.csv')
        XrecSes = Xrec;
        rmsSes = rms;
        rmsASes = rmsA;
        costASes = costA;
        var_exp_Ses = var_exp;
        accu_Ses = accu;
        figure
        scatter(SSes(1,:),SSes(2,:),'filled')
        title('Isolation')
      

     
        clear rmsA
        clear costA
        clear var_exp 
        clear Vr 
        clear accu

           elseif w == 7
         [ AFin, SFin, MuFin, VFin, cvFin, hpFin, lcFin] = pca_full( X_, zz(y-1), opts );
         Xrec = repmat(MuFin,1,dim(1)) + AFin*SFin;
        for i = 1:size(Xrec,1)
            for j = 1:size(Xrec,2)
                    Vr(i,j) = AFin(i,:) * cvFin.S{j} * AFin(i,:)' + ...
                        SFin(:,j)' * cvFin.A{i} * SFin(:,j) + ...
                    sum( sum( cvFin.S{j} .* cvFin.A{i} ) ) + cvFin.Mu(i);
            end
        end
        VrFin = Vr;
        T = array2table(Xrec');
        T.Properties.VariableNames = cols;
        writetable(T,'Finch_VBPCA_Recon_All.csv')
        XrecFin = Xrec;
        rmsFin = rms;
        rmsAFin = rmsA;
        costAFin = costA;
        var_exp_Fin = var_exp;
        accu_Fin = accu;
        figure
        scatter(SFin(1,:),SFin(2,:),'filled')
        title('Isolation')
      

     
        clear rmsA
        clear costA
        clear var_exp 
        clear Vr 
        clear accu

          elseif w == 8
         [ AGen, SGen, MuGen, VGen, cvGen, hpGen, lcGen] = pca_full( X_, zz(y-1), opts );
         Xrec = repmat(MuGen,1,dim(1)) + AGen*SGen;
        for i = 1:size(Xrec,1)
            for j = 1:size(Xrec,2)
                    Vr(i,j) = AGen(i,:) * cvGen.S{j} * AGen(i,:)' + ...
                        SGen(:,j)' * cvGen.A{i} * SGen(:,j) + ...
                    sum( sum( cvGen.S{j} .* cvGen.A{i} ) ) + cvGen.Mu(i);
            end
        end
        VrGen = Vr;
        XrecGen = Xrec;
        rmsGen = rms;
        rmsAGen = rmsA;
        costAGen = costA;
        var_exp_Gen = var_exp;
        accu_Gen = accu;
        figure
        scatter(SGen(1,:),SGen(2,:),'filled')
        title('Genetics')
      

     
        clear rmsA
        clear costA
        clear var_exp 
        clear Vr 
        clear accu

        end





