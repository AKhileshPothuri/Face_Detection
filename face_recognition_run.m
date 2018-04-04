% MAT160 Assignment 
% Face recognition system (Eigenface method)
% CREATEDATABASE, EIGENFACECORE, RECOGNITION

clear all
clc
close all

TrainDatabasePath = uigetdir('C:\Users\manju\Desktop\SNU YR_2\PCA_basedFaceRecognitionSystem', 'Select training database path' );
TestDatabasePath = uigetdir('C:\Users\manju\Desktop\SNU YR_2\PCA_basedFaceRecognitionSystem', 'Select test database path');

prompt = {'Enter test image name (a number between 1 to 10):'};
dlg_title = 'Input of Face Recognition System[Eigen Faces]';
num_lines= 1;
def = {'1'};
%1
TestImage  = inputdlg(prompt,dlg_title,num_lines,def);
%2
TestImage = strcat(TestDatabasePath,'\',char(TestImage),'.jpg');
%3
im = imread(TestImage);

T = CreateDatabase(TrainDatabasePath);
[m, A, Eigenfaces] = EigenfaceCore(T);
OutputName = Recognition(TestImage, m, A, Eigenfaces);

SelectedImage = strcat(TrainDatabasePath,'\',OutputName);
SelectedImage = imread(SelectedImage);

imshow(im)
title('Test Image');
figure,imshow(SelectedImage);
title('Equivalent Image');

str = strcat('Matched image is :  ',OutputName);
disp(str)
