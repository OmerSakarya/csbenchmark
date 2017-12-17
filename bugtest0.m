% Benchmark version (original code)
% The goal is to demonstrate that MOSEK has a memory leak.
% In this example, a rank-1 matrix is recovered from dimension deficient data

% initialize
%clc
clear
rng(1337);
%run /home/omer/cvx/cvx_setup.m
cvx_solver 'sdpt3'

% matrix dimension 
n = 60;

% number of tests
reps = 20;

% time data
t_inner = zeros(reps,1);
t_outer = 0;

t_outer = tic;
for j=1:reps
    disp(j);
    % draw rank-1 matrix and vectorize
    M0 = randn([n,1])*randn([n,1])';
    vecmat = reshape(M0,[n^2,1]);
    
    % draw linear map
    A = randn(6*n,n^2);
    
    % y = incomplete information about M0
    y = A*vecmat;
    
    %optimize
    t_inner_scalar = tic;
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
    % save elapsed time of single solver call
    t_inner(j) = toc(t_inner_scalar);
    
    % compare M to M0
    fmat(j) = norm(reshape(M,[n^2,1]) - vecmat);
end
% save time it took for all reps to complete, includes random data
% generation etc.
t_outer = toc(t_outer);

% largest deviation
largest = max(abs(fmat));
display(['largest deviation= ' num2str(largest)]);

% number of reps, time and other info
display([num2str(reps) ' reps']);

% mean time of all reconstructions with data prep overhead
display(['total time = ' num2str(t_outer)]);

% mean time of a single reconstruction without overhead
t_mean = mean(t_inner);
display(['mean time ' num2str(t_mean)]);

% standard deviation
t_standard_deviation = std(t_inner);
display(['time standard deviation ' num2str(t_standard_deviation)]);

