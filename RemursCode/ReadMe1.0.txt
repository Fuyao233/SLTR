%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                        Matlab source codes for                              %
%             Regularized Multilinear Regression and Selection (Remurs)       %
%                                                                             %
% Author: Xiaonan Song, Wenwen Li, and Haiping Lu*                            %
% *Email : hplu@ieee.org   or   eehplu@gmail.com  or h.lu@sheffield.ac.uk     %
% Release date: 26th March 2019                                               %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[Algorithms]%
The matlab codes provided here implement the algorithms presented in the 
paper "Remurs_AAAI17.pdf" included in this package with directory of Paper/Remurs_AAAI17.pdf:

	Song, Xiaonan and Lu, Haiping, 
	"Multilinear regression for embedded feature selection with application to fmri analysis", 
	Thirty-First AAAI Conference on Artificial Intelligence (AAAI), pp. 2562-2568, 2017.


Algorithm 1: "Code/Remurs.m" implements the Remurs algorithm described in this paper.
             "Code/main_Remurs.m" implements an example of using Remurs on rescaled CMU data.				
Algorithm 2: "Codes/main_lowRank.m" implements the version of Remurs with only tensor nuclear norm but not L1 norm in this paper
Algorithm 3: "Code/main_lasso.m" implements Lasso regularised linear regression model
Algorithm 4: "Code/main_ENet.m" implements ENet regularised linear regression model
Algorithm 5: "Code/main_ridge.m" implements Ridge regularised linear regression model
---------------------------

%[Data]%
The dataset used in this code is CMU2008 dataset (Mitchell et al. 2008). It aims to predict human brain activity associated with the meanings of nouns. 

    Mitchell, Tom M and Shinkareva, Svetlana V and Carlson, Andrew and Chang, Kai-Min and Malave, Vicente L and Mason, Robert A and Just, Marcel Adam
    "Predicting human brain activity associated with the meanings of nouns"
    Science, 320(5880), pp.1191-1195.

The full dataset download link: http://www.cs.cmu.edu/afs/cs/project/theo-73/www/science2008/data.html

%[Usages]%
Please refer to the comments in the codes, which include example usage on one of the recaled subject (subject 1) from CMU dataset:

Data/CMU/RescaledP1.mat: 3D SPM fMRI data of size 16 × 19 × 7 (2,128 voxels). We focus on binary classification of "animals" vs. "tool". 
                              The class "animals" combines observations from "animal" and "insect", and the class "tools" combines "tool" and "furniture" in the CMU dataset. 
                              Thus, there are 120 observations for each class.
---------------------------

%[Brain decoding testing and results verification]%
You can test the code on the rescaled data with the following steps:

1. Run main_remurs.m to get the results for Remurs
2. Run main_lowRank.m to get the results for Remurs with only considering tensor nuclear norm
3. Run main_lasso.m to get results for lasso
4. Run main_ENet.m to get results for ENet
5. Run main_ridge.m to get results for Ridge

Please note that the results presented in Table I of the AAAI17 paper were obtained with the CMU2008 dataset at the resolution 51x61x23 on Matlab R2016a. To verify the results, you need to download the original data via the link about and change the input data. Slightly different results may be obtained with the latest Matlab versions. 
---------------------------

%[Toolbox needed]%:
This code needs the tensor toolbox available at https://www.sandia.gov/~tgkolda/TensorToolbox/index-2.1.html 
This code also requires SLEP pakcage which are available at https://github.com/jiayuzhou/SLEP
Please download these two packages and put them under "Code", i.e., Code/SLEP_package_4.1 and Code/tensor_toolbox_2.1
---------------------------

%[Restriction]%
In all documents and papers reporting research work that uses the matlab codes 
provided here, the respective author(s) must reference the following paper: 

[1] Song, Xiaonan and Lu, Haiping, 
	"Multilinear Regression for Embedded Feature Selection with Application to fMRI Analysis", 
	Thirty-First AAAI Conference on Artificial Intelligence (AAAI), pp. 2562-2568, 2017.
---------------------------

%[Contact]%
Email: hplu@ieee.org or eehplu@gmail.com
---------------------------

%[Update history]%

1. March 26, 2019: Version 1.0 is released 
[The original code was written by Xiaonan Song in Matlab R2015a, which was modified by Haiping Lu and Wenwen Li for this release tested with Matlab R2018a).]