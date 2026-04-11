function label_pred = sc_wgb(data_matrix, cluster_num, wgb_num, options)
% SC_WGB  基于加权粒球的谱聚类
%
% 输入:
%   data_matrix  - (instance_num, feature_num) 数据矩阵
%   cluster_num  - 聚类数量
%   wgb_num      - 加权粒球数量
%   options      - 算法参数
%
% 输出:
%   label_pred   - 样本预测标签

%% (0) 参数初始化
arguments
    data_matrix (:,:) double {mustBeNonempty, mustBeFinite}
    cluster_num (1,1) double {mustBeInteger, mustBePositive}
    wgb_num (1,1) double {mustBeInteger, mustBePositive} = fix(sqrt(size(data_matrix,1)))+1
    options (1,1) wgbOptions = wgbOptions()
end

knn_num = max(3, ceil(log(wgb_num)));

%% (1) 生成加权粒球
outputs = generate_wgbs(data_matrix, wgb_num, options);

center_matrix = gather(outputs.center_matrix);
feature_weight = gather(outputs.feature_weight);
membership_matrix = gather(outputs.membership_matrix);

clear outputs

%% (2) 计算粒球之间的加权平方欧氏距离
distance_matrix_gb = weighted_squared_euclidean_distance( ...
    center_matrix, center_matrix, feature_weight);

clear data_matrix center_matrix feature_weight

%% (3) 构建 KNN 图
[min_value, ~] = mink(distance_matrix_gb, knn_num, 2);

distance_matrix_gb(distance_matrix_gb > min_value(:,end)) = inf;

%% (4) Gaussian Kernel
kernel_width = sqrt(mean(min_value(:,end)));

kernel_width = max(kernel_width, eps);

similarity_matrix = exp(-distance_matrix_gb ./ (2*kernel_width^2));

similarity_matrix(isinf(distance_matrix_gb)) = 0;

%% (5) 转为稀疏矩阵并对称化
similarity_matrix = sparse(similarity_matrix);

similarity_matrix = (similarity_matrix + similarity_matrix') / 2;

%% (6) 谱聚类
normalization_methods = {'randomwalk','symmetric','none'};

label_gbs = [];

for i = 1:length(normalization_methods)

    try

        label_gbs = spectralcluster( ...
            similarity_matrix, ...
            cluster_num, ...
            'LaplacianNormalization', normalization_methods{i});

        break

    catch

        continue

    end

end

if isempty(label_gbs)
    error("Spectral clustering failed.");
end

%% (7) 粒球标签 → 样本标签

[~, idx] = max(membership_matrix, [], 2);

label_pred = label_gbs(idx);

end