% Benchmark version (Yalmip and nuclear norm code)
% The goal is to demonstrate that MOSEK has a memory leak.
% In this example, a rank-1 matrix is recovered from dimension deficient data

% initialize
clear

% matrix dimension 
n = 60;

% number of tests
reps = 20;

% PRNG seed
randn("seed", 1337);

% set solver here
solver = 'csdp';

% following function was used for some multithreading

%function t = maxNumCompThreads()
%  t = 8;
%endfunction

tic% time data
t_inner = zeros(reps,1);
t_outer = 0;

t_outer = tic;
for j=1:reps
    disp(j);
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
    t_inner_scalar = tic;
    diagnostics = optimize(constraints, objective, options);
    t_inner(j) = toc(t_inner_scalar)
    solution = value(M);
    
    % compare M to M0
    fmat(j) = norm(reshape(solution,[n^2,1]) - vecmat);
    M0(1,1:5)
    solution(1,1:5)
end
% save time it took for all reps to complete, includes random data
% generation etc.
t_outer = toc(t_outer);

disp("\n\n");
disp(diagnostics);
disp("\n\n");

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

