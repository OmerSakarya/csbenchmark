% Benchmark version (original code)
% The goal is to demonstrate that MOSEK has a memory leak.
% In this example, a rank-1 matrix is recovered from dimension deficient data

% initialize
%clc
clear
rng(1336);
run /home/omer/cvx/cvx_setup.m
cvx_solver 'sdpt3'

% matrix dimension 
n = 60;

% number of tests
reps = 1;

tic
for j=1:reps
    % draw rank-1 matrix and vectorize
    M0 = randn([n,1])*randn([n,1])';
    vecmat = reshape(M0,[n^2,1]);
    
    % draw linear map
    A = randn(6*n,n^2);
    
    % y = incomplete information about M0
    y = A*vecmat;
    
    %optimize
    tic
    cvx_begin sdp
        variable X(n,n)
        variable Y(n,n)
        variable M(n,n)
        
        % minimize trace norm of M
        minimize(trace(X+Y)/2)
        [(X+X')/2 M;
         M' (Y+Y')/2] >= 0
        
        % subject to y == A*M
        y == A*reshape(M,[n^2,1])
    cvx_end
    cvx_clear
    toc
    
    % compare M to M0
    fmat(j) = norm(reshape(M,[n^2,1]) - vecmat);
end
toc

% largest deviation
largest = max(abs(fmat));
display(['largest deviation= ' num2str(largest)]);
