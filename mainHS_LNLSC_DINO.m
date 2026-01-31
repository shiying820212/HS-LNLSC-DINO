% This is an example code for running the ScSPM algorithm described in "Linear 
% Spatial Pyramid Matching using Sparse Coding for Image Classification" (CVPR'09)
%
% Written by Jianchao Yang @ IFP UIUC
% For any questions, please email to jyang29@ifp.illinois.edu.
% 
% Revised May, 2010 by Jianchao Yang
clear all;
clc;
%% parameter setting
% img_dir = 'image';                  % directory for dataset images
% data_dir = 'data';                  % directory to save the sift features of the chosen dataset
% dataSet = 'Corel10';

% img_dir = 'imageMID';                  % directory for dataset images
% data_dir = 'dataMID';                  % directory to save the sift features of the chosen dataset
% %dataSet = 'MID';

img_dir = 'imageSeaShips';                  % directory for dataset images
data_dir = 'dataSeaShips';                  % directory to save the sift features of the chosen dataset
dataSet = 'SeaShips';

% img_dir = 'imageSeaShipClip';                  % directory for dataset images
% data_dir = 'dataSeaShipClip';                  % directory to save the sift features of the chosen dataset
% dataSet = 'SeaShipsClip';

% img_dir = 'imageSMDClip';                  % directory for dataset images
% data_dir = 'dataSMDClip';                  % directory to save the sift features of the chosen dataset
% dataSet = 'SMDClip';



img_dir_List=["imageSeaShips","imageScene15","imageSeaShipClip","imageSMDClip","imageCorel10","imageCaltech101","imageCaltech256"];
data_dir_List=["dataSeaShips","dataScene15","dataSeaShipClip","dataSMDClip","dataCorel10","dataCaltech101","dataCaltech256"];
dataSet_List=["SeaShips","Scene15","SeaShipsClip","SMDClip","Corel10","Caltech101","Caltech256"];
imgTrainList=[5,];

% img_dir_List=["imageSeaShips",];
% data_dir_List=["dataSeaShips",];
% dataSet_List=["SeaShips",];

componentList=["SIFT","VitCls","VitPatch","SIFTVitCls","SIFTVitPatch"];
img_dir_len=length(img_dir_List);
componentTrainList=[4,];

%% set path
addpath('large_scale_svm');
addpath('sift');
addpath(genpath('sparse_coding'));
% sift descriptor extraction
skip_cal_sift = true;              % if 'skip_cal_sift' is false, set the following parameter
gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;                 % low contrast region normalization threshold (descriptor length)

% dictionary training for sparse coding
skip_dic_training = false;
nBases = 1024;
nsmp = 5000;
beta = 1e-5;                        % a small regularization for stablizing sparse coding
num_iters = 5;

% feature pooling parameters
pyramid = [1, 2, 4];                % spatial block number on each level of the pyramid
gamma = 0.15;
knn = 200;                          % find the k-nearest neighbors for approximate sparse coding
                                    % if set 0, use the standard sparse coding
pooling='maxpooling';

% classification test on the dataset
nRounds = 5;                        % number of random tests
lambda = 0.1;                       % regularization parameter for w
tr_num =50;% 50;                        % training number per category
tr_num_list=[50,100,150,200,250];


% for datasetIndex = 7:img_dir_len        %不同数据集
for datasetIndex = imgTrainList
    %["SeaShips","Scene15","SeaShipsClip","SMDClip","Corel10","Caltech101","Caltech256"]
    if datasetIndex==1
%         tr_num_list=[50,100,150,200,250];
        tr_num_list=[15,30,45,60];
    elseif datasetIndex==2
        tr_num_list=[15,30,45,60];       %要根据不同数据集更改
    elseif datasetIndex==3
%         tr_num_list=[50,100,150,200,250];
        tr_num_list=[15,30,45,60];
    elseif datasetIndex==4
%         tr_num_list=[50,100,150,200,250];
        tr_num_list=[15,30,45,60];
    elseif datasetIndex==5
%         tr_num_list=[15,30,45,60];
        tr_num_list=[50,];
    elseif datasetIndex==6
        tr_num_list=[15,30];
    elseif datasetIndex==7
        tr_num_list=[15,30,45,60];
    end
    img_dir = img_dir_List(datasetIndex);                  % directory for dataset images
    data_dir = data_dir_List(datasetIndex);                  % directory to save the sift features of the chosen dataset
    dataSet = dataSet_List(datasetIndex);
    
    rt_img_dir = fullfile(img_dir, dataSet);
    rt_data_dir = fullfile(data_dir, dataSet);
    
    if skip_cal_sift,
        database = retr_database_dir(rt_data_dir);
    else
        [database, lenStat] = CalculateSiftDescriptor(rt_img_dir, rt_data_dir, gridSpacing, patchSize, maxImSize, nrml_threshold);
    end;
    %% load sparse coding dictionary (one dictionary trained on Caltech101 is provided: dict_Caltech101_1024.mat)
    
    X = rand_sampling(database, nsmp);
    [B, S, stat] = reg_sparse_coding(X, nBases, eye(nBases), beta, gamma, num_iters);
    nBases = size(B, 2);                    % size of the dictionary
    
    %% calculate the sparse coding feature

    for componentIndex = componentTrainList     %不同组合
        %['SIFT','VitCls','VitPatch','SIFTVitCls','SIFTVitPatch'];
%         layerList=[2,4,6,8,10,12];
        layerList=[12,];        %只跑12层
        layerLen=length(layerList);
        if componentIndex==1
            dimFea = sum(nBases*pyramid.^2);
            layerList=[2,];
            layerLen=length(layerList);
        elseif componentIndex==2
            dimFea=768;
        elseif componentIndex==3
            dimFea=128*400;
        elseif componentIndex==4
            dimFea = sum(nBases*pyramid.^2)+768;
        elseif componentIndex==5
            dimFea = sum(nBases*pyramid.^2)+128*400;
        end
        % dimFea=768;
%         dimFea = sum(nBases*pyramid.^2)+768;
        numFea = length(database.path);
        
        sc_fea = zeros(dimFea, numFea);
        sc_label = zeros(numFea, 1);
        
        disp('==================================================');
        fprintf('Calculating the sparse coding feature...\n');
        fprintf('Regularization parameter: %f\n', gamma);
        fprintf('Pooling method: %s\n',pooling);
        disp('==================================================');
        
        for layerIndex=1:layerLen
            nowLayer=layerList(layerIndex);     %不同层
            for iter1 = 1:numFea,  
                if ~mod(iter1, 50),
                    fprintf('.\n');
                else
                    fprintf('.');
                end;
                fpath = database.path{iter1};
                %['SIFT','VitCls','VitPatch','SIFTVitCls','SIFTVitPatch'];
                if componentIndex==1
                    load(fpath);
                    sc_fea(:, iter1) = sc_approx_pooling(feaSet, B, pyramid, gamma, knn);     %原始方法
                elseif componentIndex==2
                    vitPath = strrep(fpath, data_dir, data_dir+"VitCls"+string(nowLayer));
                    load(vitPath);
                    sc_fea(:, iter1) = feaSetVit.feaArr(:);    %cls token 768
                elseif componentIndex==3
                    vitPatchPath = strrep(fpath, data_dir, data_dir+"VitPatch"+string(nowLayer));
                    load(vitPatchPath);
                    sc_fea(:, iter1) = feaSetVitPatch.feaArr(:);    %patch token 128*400
                elseif componentIndex==4
                    load(fpath);
                    vitPath = strrep(fpath, data_dir, data_dir+"VitCls"+string(nowLayer));
                    load(vitPath);
                    siftF= sc_approx_pooling(feaSet, B, pyramid, gamma, knn);     %原始方法
                    sc_fea(:, iter1)=[siftF; feaSetVit.feaArr(:)];
                elseif componentIndex==5
                    load(fpath);
                    vitPatchPath = strrep(fpath, data_dir, data_dir+"VitPatch"+string(nowLayer));
                    load(vitPatchPath);
                    siftF = sc_approx_pooling(feaSet, B, pyramid, gamma, knn);     %原始方法
                    sc_fea(:, iter1)=[siftF; feaSetVitPatch.feaArr(:)];
                end
                
%                 betaCls = vitFea.feaArr(:);    %cls token 768
%                 sc_fea(:, iter1)=beta;
           
                sc_label(iter1) = database.label(iter1);
            end;
            
            trainLevels = length(tr_num_list);  %不同样本数
            for trianIndex = 1:trainLevels
                tr_num=tr_num_list(trianIndex);
                
                %% evaluate the performance of the computed feature using linear SVM      
                [dimFea, numFea] = size(sc_fea);
                clabel = unique(sc_label);
                nclass = length(clabel);
                
                accuracy = zeros(nRounds, 1);
                %["SeaShips","Scene15","SeaShipsClip","SMDClip","Corel10","Caltech101","Caltech256"]
                if datasetIndex==1
                    accClass= zeros(6, nRounds);        %类数目
                elseif datasetIndex==2
                    accClass= zeros(15, nRounds);        %类数目
                elseif datasetIndex==3
                    accClass= zeros(6, nRounds);        %类数目
                elseif datasetIndex==4
                    accClass= zeros(9, nRounds);        %类数目
                elseif datasetIndex==5
                    accClass= zeros(10, nRounds);        %类数目
                elseif datasetIndex==6
                    accClass= zeros(102, nRounds);        %类数目
                elseif datasetIndex==7
                    accClass= zeros(257, nRounds);        %类数目
                end
    %             accClass= zeros(6, nRounds);        %类数目
                last_fea = [];
                
                for ii = 1:nRounds,
                    fprintf('Round: %d...\n', ii);
                    tr_idx = [];
                    ts_idx = [];
                    
                    for jj = 1:nclass,
                        idx_label = find(sc_label == clabel(jj));
                        num = length(idx_label);
                        
                        idx_rand = randperm(num);
                        
                        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
                        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
                    end;
                    
                    tr_fea = sc_fea(:, tr_idx);
                    tr_label = sc_label(tr_idx);
                    
                    ts_fea = sc_fea(:, ts_idx);
                    ts_label = sc_label(ts_idx);
                    
                    [w, b, class_name] = li2nsvm_multiclass_lbfgs(tr_fea', tr_label, lambda);
                
                    %[C, Y] = li2nsvm_multiclass_fwd(ts_fea', w, b, class_name);
                    [~, Y] = li2nsvm_multiclass_fwd(sc_fea', w, b, class_name);
                    last_fea = [last_fea,Y];
                end;
                
                last_fea = last_fea';
                for ii = 1:nRounds
                    fprintf('Round: %d...\n', ii);
                    tr_idx = [];
                    ts_idx = [];
                    
                    for jj = 1:nclass,
                        idx_label = find(sc_label == clabel(jj));
                        num = length(idx_label);
                        
                        idx_rand = randperm(num);
                        
                        tr_idx = [tr_idx; idx_label(idx_rand(1:tr_num))];
                        ts_idx = [ts_idx; idx_label(idx_rand(tr_num+1:end))];
                    end;
                    
                    tr_fea = last_fea(:, tr_idx);
                    tr_label = sc_label(tr_idx);
                    
                    ts_fea = last_fea(:, ts_idx);
                    ts_label = sc_label(ts_idx);
                    
                    [w, b, class_name] = li2nsvm_multiclass_lbfgs(tr_fea', tr_label, lambda);
                
                    [C, Y] = li2nsvm_multiclass_fwd(ts_fea', w, b, class_name);
                
                    acc = zeros(length(class_name), 1);
                    
                    for jj = 1 : length(class_name),
                        c = class_name(jj);
                        idx = find(ts_label == c);
                        curr_pred_label = C(idx);
                        curr_gnd_label = ts_label(idx);    
                        acc(jj) = length(find(curr_pred_label == curr_gnd_label))/length(idx);
                    end; 
                    
                    accuracy(ii) = mean(acc); 
                    accClass(:,ii) = acc; 
                end;
                fprintf('Mean accuracy: %f\n', mean(accuracy));
                fprintf('Standard deviation: %f\n', std(accuracy));
                accSavePath="saves/"+"Acc_"+dataSet+"_"+componentList(componentIndex)+"_"+"L"+string(nowLayer)+"_"+string(tr_num)+".mat";
                AccRst.accuracy = accuracy;
                AccRst.accClass=accClass;
                AccRst.mean=mean(accuracy);
                AccRst.std=std(accuracy);
                save(accSavePath, 'AccRst');
            end
        end
    end
end


