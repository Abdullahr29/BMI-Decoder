function [eigenvalues, principal_components] = our_pca(data, variance_explained)
    [U,S,~] = svd(data, "econ");
    eigenvalues = diag(S);
    principal_components = U;
    total = 0;
    
    if(variance_explained == 1)
        for i=1:size(data)
            total = total + eigenvalues(i);
        end
        eigenvalues = (eigenvalues)./total;
        for i=2:size(data)
            eigenvalues(i) =eigenvalues(i-1)+ eigenvalues(i);
        end
    end
end
