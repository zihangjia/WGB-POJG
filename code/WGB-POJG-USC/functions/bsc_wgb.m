function label_pred = bsc_wgb(data_matrix, cluster_num, wgb_num, options)
%BSC_WGB 实现基于加权粒球的二部图谱聚类
% 输入:
%   data_matrix - 数据矩阵 (instance_num*feature_num) (非空)
%   cluster_num - 簇的数量 (1,1) (正整数)
%   wgb_num - 投影粒球的数量 (1,1) (正整数, 默认=sqrt(instance_num))
%   options - 算法的运行参数
% 输出:
%   label_pred - 样本标签 (instance_num*1)

% (0) 参数检验与初始化 ******************************************************
arguments
    data_matrix (:,:) double {mustBeNonempty, mustBeFinite}
    cluster_num (1,1) double {mustBeInteger, mustBePositive}
    wgb_num (1,1) double {mustBeInteger, mustBePositive} = fix(sqrt(size(data_matrix, 1))) + 1
    options (1,1) wgbOptions = wgbOptions()
end
knn_num = 5; 

% (1) 生成投影粒球 ******************************************************
outputs = generate_wgbs(data_matrix, wgb_num, options);
center_matrix = gather(outputs.center_matrix);
feature_weight = gather(outputs.feature_weight);
clear outputs
% center_matrix 和 feature_weight_vector 提取到内存中


% (2) 计算样本与粒球的特征加权平方欧氏距离矩阵 ********************************
distance_matrix_instances_with_gbs = weighted_squared_euclidean_distance(data_matrix, center_matrix, feature_weight);
clear data_matrix center_matrix feature_weight
distance_matrix_instances_with_gbs = sqrt(distance_matrix_instances_with_gbs);
% —— distance_matrix_instances_with_gbs - (instance_num, wgb_num) 样本与粒球球心的加权欧氏距离矩阵


% ***************** 建立样本与粒球的相似性矩阵 *****************
[min_value, ~] = mink(distance_matrix_instances_with_gbs, knn_num, 2);
distance_matrix_instances_with_gbs(distance_matrix_instances_with_gbs > min_value(:, end)) = inf;
kernel_width = max(mean(min_value(:, end), 'all'), eps);
similarity_matrix_instances_with_gbs = exp(-(distance_matrix_instances_with_gbs.^2) ./ (2*kernel_width^2));
similarity_matrix_instances_with_gbs(isinf(distance_matrix_instances_with_gbs)) = 0;
similarity_matrix_instances_with_gbs = sparse(similarity_matrix_instances_with_gbs);
% (1) 计算样本与其最近的 knn_num 个粒球的距离
% (2) 将与样本距离大于 knn_num 的粒球的距离置为无穷大
% (3) 计算高斯核的宽度
% (4) 计算样本与粒球的相似性
% (5) 计算相似性矩阵转换为系数矩阵
% ***************** 建立样本与粒球的相似性矩阵 *****************

% *****************   使用转移切获得聚类结果 *****************
label_pred = transfer_cut_for_bipartite_graph(similarity_matrix_instances_with_gbs, cluster_num);
% *****************   使用转移且获得聚类结果 *****************
end

function label_pred = transfer_cut_for_bipartite_graph(bipartite_graph, sub_graph_num, max_iterations, k_means_repetitions)
%transfer_cut_for_bipartite_graph 采用转移切分割二部图
% bipartite_graph, node_num_1*node_num_2, 二部图的相似性矩阵
% sub_graph_num, 1*1, 分割得到的子图的数量
% max_iterations, 1*1, k-means 最大执行次数
% k_means_repetitions, 1*1, k-means 重复次数

% ************************ 算法初始化 ************************
if nargin < 3
    max_iterations = 100;
end
if nargin < 4
    k_means_repetitions = 5;
end
[node_num_1, node_num_2] = size(bipartite_graph);

% (1)-(3) 设置 k-means 最大执行次数为 100
% (4)-(6) 设置 k-means 重复执行次数为 5
% ************************ 算法初始化 ************************


% ************** 计算小规模图的仿射矩阵 **************
degree_vector_part_1 = sum(bipartite_graph, 2);
degree_vector_part_1(degree_vector_part_1==0) = eps;
degree_matrix_part_1_reciprocal = sparse(1:node_num_1, 1:node_num_1, 1 ./ (degree_vector_part_1 + 1e-10));
affinity_matrix = bipartite_graph' * degree_matrix_part_1_reciprocal * bipartite_graph;
% (1) 计算二部图第 1 部分的节点的度向量
% (2) 计算小规模图的仿射矩阵
% ************** 计算小规模图的仿射矩阵 **************

% ************** 计算小规模图的归一化切 **************
degree_vector = sum(affinity_matrix, 2);
degree_matrix_reciprocal = sparse(1:node_num_2, 1:node_num_2, 1./(sqrt(degree_vector) + 1e-10));
normalized_affinity_matrix = degree_matrix_reciprocal * affinity_matrix * degree_matrix_reciprocal;
normalized_affinity_matrix = (normalized_affinity_matrix + normalized_affinity_matrix') / 2;
[eigen_matrix, ~] = eigs(normalized_affinity_matrix, sub_graph_num);
normalized_cut_eigen_matrix = degree_matrix_reciprocal * eigen_matrix;
eigen_matrix = degree_matrix_part_1_reciprocal * bipartite_graph * normalized_cut_eigen_matrix;
eigen_matrix = eigen_matrix ./ (vecnorm(eigen_matrix,2,2) + 1e-10);
% (1) 计算小图的度向量
% (2) 计算度矩阵的倒数 (已开平方) 并转换为稀疏矩阵
% (3) 计算归一化的仿射矩阵
% (4) 确保归一化矩阵是对称矩阵
% (5) 对归一化小图做快速特征分解
% (6) 计算归一化切的特征矩阵
% (7) 计算大图的特征矩阵
% (8) 对大图的特征矩阵做归一化
% ************** 计算小规模图的归一化切 **************



% *************** 对特征矩阵进行聚类得到聚类结果 ***************
label_pred = kmeans(eigen_matrix, sub_graph_num, MaxIter=max_iterations, Replicates=k_means_repetitions);
% *************** 对特征矩阵进行聚类得到聚类结果 ***************
end