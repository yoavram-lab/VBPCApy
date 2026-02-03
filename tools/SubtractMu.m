function [X,Xprobe] = SubtractMu( Mu, X, M, Xprobe, Mprobe, update_bias )

n2 = size(X,2);

if ~update_bias
    return
end
   
if issparse(X)
    X = subtract_mu( X, Mu );
    if ~isempty(Xprobe)
        Xprobe = subtract_mu( Xprobe, Mu );
    end
else
    X = X - repmat(Mu,1,n2).*M;
    if ~isempty(Xprobe)
        Xprobe = Xprobe - repmat( Mu, 1, n2 ).*Mprobe;
    end
end
