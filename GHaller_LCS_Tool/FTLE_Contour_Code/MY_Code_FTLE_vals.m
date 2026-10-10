%% Input parameters
epsilon = .1;
amplitude = .1;
omega = pi/10;
T = 2*pi/omega;
timespan = [0,-T];
domain = [0,2;0,1];
resolutionX = 200;
[resolutionY,deltaX] = equal_resolution(domain,resolutionX);
resolution = [resolutionX,resolutionY];

%% Velocity definition
lDerivative = @(t,x,~)derivative(t,x,false,epsilon,amplitude,omega);
incompressible = true;

%% LCS parameters
% Cauchy-Green strain
cgStrainOdeSolverOptions = odeset('relTol',1e-5);

hAxes = setup_figure(domain);
title(hAxes,'Repelling and elliptic LCSs')

%% Cauchy-Green strain eigenvalues and eigenvectors
[cgEigenvector,cgEigenvalue] = eig_cgStrain(lDerivative,domain,resolution,timespan,'incompressible',incompressible,'odeSolverOptions',cgStrainOdeSolverOptions);

%% Plotting FTLE values

% Extracting FTLE values
FTLE = ftle(cgEigenvalue(:,1),diff(timespan));

% Plotting FTLE
plot_ftle(hAxes,domain,resolution,FTLE);