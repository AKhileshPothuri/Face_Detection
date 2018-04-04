function [m, A, Eigenfaces] = EigenfaceCore(T)

% Description: This function gets a 2D matrix, containing all training image vectors
% and returns 3 outputs which are extracted from training database.
%
%  T                                    - A 2D matrix, containing all 1D image vectors.
%                                         Suppose all P images in the training database 
%                                         have the same size of MxN. So the length of 1D 
%  .                                       column vectors is M*N and 'T' will be a MNxP 2D matrix.
% .
% Returns:       m                      - (M*Nx1) Mean of the training database
%                Eigenfaces             - (M*Nx(P-1)) Eigen vectors of the covariance matrix of the training database
%                A                      - (M*NxP) Matrix of centered image vectors
            
% Calculating the mean image 
m = mean(T,2);% Computing the average face image m = (1/P)*sum(Tj's)    (j = 1 : P); 2 is the dim of matrix T
Train_Number = size(T,2);

% Calculating the deviation of each image from mean image
A = [];  
for i = 1 : Train_Number
    temp = double(T(:,i)) - m; % Computing the difference image for each image[each column in T] in the training set Ai = Ti - m
    A = [A temp]; % Merging all centered images
end

L = A'*A; % L is the surrogate of covariance matrix C=A*A'.
[V D] = eig(L); % produces a diagonal matrix D of eigenvalues and a full matrix V whose columns are the corresponding eigenvectors

% Sorting and eliminating eigenvalues
% All eigenvalues of matrix L are sorted and those who are less than a
% specified threshold, are eliminated. So the number of non-zero
% eigenvectors may be less than (P-1).

L_eig_vec = [];
for i = 1 : size(V,2) 
    if( D(i,i)>1 )%The M eigenvalues of A'A (along with their corresponding eigenvectors) correspond to the M largest eigenvalues of AA' (along
                  % with their corresponding eigenvectors).
        L_eig_vec = [L_eig_vec V(:,i)];
    end
end

% Eigenvectors of covariance matrix C (or so-called "Eigenfaces")
% can be recovered from L's eiegnvectors.
Eigenfaces = A * L_eig_vec; % A: centered image vectors
display(size(A));
display(size(L_eig_vec));



