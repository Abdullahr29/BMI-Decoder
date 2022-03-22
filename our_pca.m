 function [eigenvalues, principal_components] = our_pca(data, variance_explained, dims)
    % data: features/variables in columns, samples/timesteps in rows
    [~,S,V] = svd(data);
    eigenvalues = diag(S);
    principal_components = V;
    total = 0;
    
    [~, features] = size(data);
    
    if(variance_explained == 1)
        for i=1:features
            total = total + eigenvalues(i);
        end
        eigenvalues = (eigenvalues)./total;
        for i=2:features
            eigenvalues(i) =eigenvalues(i-1)+ eigenvalues(i);
        end
    end
    principal_components = principal_components(:,1:dims);
end
