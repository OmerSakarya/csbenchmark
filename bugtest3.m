% Benchmark version (Yalmip and nuclear norm code)
% The goal is to demonstrate that MOSEK has a memory leak.
% In this example, a rank-1 matrix is recovered from dimension deficient data

% initialize
clear

% matrix dimension 
n = 60;

% number of tests
reps = 1;

% PRNG seed
randn("seed", 1337);

% set solver here
solver = 'sedumi';

% following function was used for some multithreading

%function t = maxNumCompThreads()
%  t = 8;
%endfunction

tic
for j=1:reps
    % draw rank-1 matrix and vectorize
    M0 = randn([n,1])*randn([n,1])';
    vecmat = vec(M0);
    
    % draw linear map
    A = randn(6*n,n^2);
    
    % y = incomplete information about M0
    y = A*vecmat;
    
    % optimize
    M = sdpvar(n,n,'full')
    objective = norm(M,'nuclear')
    constraints = [y == A*reshape(M,[n^2,1])]
    options = sdpsettings('solver',solver);
    tic
    diagnostics = optimize(constraints, objective, options);
    toc
    solution = value(M);
    
    % compare M to M0
    fmat(j) = norm(reshape(solution,[n^2,1]) - vecmat);
    M0(1,1:5)
    solution(1,1:5)
end
toc

disp("\n\n");
disp(diagnostics);
disp("\n\n");

% largest deviation
largest = max(abs(fmat));
display(['largest deviation= ' num2str(largest)]);


