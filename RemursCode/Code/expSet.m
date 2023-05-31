function [info]=expSet
info.classes = 2; % 1:binary-2 classes 2:multi-4 classes , 2,,6,82,,6,8[4]
info.subjects = [1 2 3 4 5 6 7 8 9]; %
info.alphaRange = [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2];
%info.alphaRange = [0.1];
 
info.betaRange  = [10^-3, 5*10^-3, 10^-2, 5*10^-2, 10^-1, 5*10^-1, 10^0, 5*10^0, 10^1, 5*10^1, 10^2, 5*10^2, 10^3];
info.paraRange = [0, 10^-3, 10^-2, 10^-1, 10^0, 10^1, 10^2, 10^3];
%info.betaRange  = [0.1];
%info.paraRange = [0.1];
%info.testFold = 6;
info.valFold = 5;
info.valFold = 1;
info.numTest = 20; %120/
info.epsilon = 1e-4;

info.minFeaPercent = 0.05;
info.maxFeaPercent = 0.500;

