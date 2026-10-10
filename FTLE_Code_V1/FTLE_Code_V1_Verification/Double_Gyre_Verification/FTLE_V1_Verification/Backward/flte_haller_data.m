clc
clear

dat = importdata("data_rev.mat");

maxEig = max(dat, [], 2);

FTLE = log(maxEig)/40;

resolutionX = 200;
resolutionY = 101;
domain = [0,2;0,1];

x = linspace(domain(1,1),domain(1,2),resolutionX);

y = linspace(domain(2,1),domain(2,2),resolutionY);

[X,Y] = meshgrid(x,y);

xr = X(:);
yr = Y(:);

points = [xr yr];

data = [xr yr FTLE];
writematrix(data, "Hallerdata.csv")