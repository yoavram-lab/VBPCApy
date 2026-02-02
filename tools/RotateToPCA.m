%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Find the PCA rotation: This has to be checked
function [ dMu, A, Av, S, Sv ] = ...
    RotateToPCA( A, Av, S, Sv, Isv, obscombj, update_bias );

n1 = size(A,1);
n2 = size(S,2);

if update_bias
    mS = mean(S,2);
    dMu = A*mS;
    S = S - repmat(mS,1,n2);
else
    dMu = 0;
end

covS = S*S';
if isempty(Isv)
    for j = 1:n2
        covS = covS + Sv{j};
    end
else
    nobscomb = length(obscombj);
    for j = 1:nobscomb
        covS = covS + ( length(obscombj{j})*Sv{j} );
    end
end
    
covS = covS / n2;
%covS = covS / (n2-n1);
[VS,D] = eig(covS);
RA = VS*sqrt(D);
A = A*RA;
covA = A'*A;
if ~isempty(Av)
    for i = 1:n1
        Av{i} = RA'*Av{i}*RA;
        covA = covA + Av{i};
    end
end
covA = covA / n1;
[VA,DA] = eig(covA);
[DA,I] = sort( -diag(DA) );
DA = -DA;
VA = VA(:,I);
A = A*VA;

if ~isempty(Av)
    for i = 1:n1
        Av{i} = VA'*Av{i}*VA;
    end
end
R = VA'*diag(1./sqrt(diag(D)))*VS';

S = R*S;
for j = 1:length(Sv)
    Sv{j} = R*Sv{j}*R';
end
