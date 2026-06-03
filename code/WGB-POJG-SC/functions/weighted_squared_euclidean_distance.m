function distance_matrix = weighted_squared_euclidean_distance(data_matrix_1, data_matrix_2, feature_weight_vector)
%weighted_squared_euclidean_vector 输入 2 个样本观测矩阵, 返回样本之间的距离矩阵
% 输入:
%   data_matrix_1 - (instance_num_1, feature_num)
%   data_matrix_2 - (instance_num_2, feature_num)
%   feature_weight_vector - (feature_num, 1)
% 输出:
%   distance_matrix - (instance_num_1, instance_num_2)

feature_weight_vector = feature_weight_vector(:); 
weighted_data_1 = data_matrix_1 .* feature_weight_vector';
weighted_data_2 = data_matrix_2 .* feature_weight_vector';
weighted_square_1 = sum(weighted_data_1 .^ 2, 2);
weighted_square_2 = sum(weighted_data_2 .^ 2, 2);
cross_term = weighted_data_1 * weighted_data_2';
% (1) 确保 feature_weight_vector 是列向量
% (3) weighted_square_1, instance_num_1*1, weighted_square_1(i) =
%     sum_{k=1}^{feature_num}w_{k}^{2}data_matrix_1(i,k)^{2}
% (4) weighted_square_2, 1*instance_num_2, weighted_square_2(j) =
%     sum_{k=1}^{feature_num}w_{k}^{2}data_matrix_2(j,k)^{2}
% (5) cross_term, instance_num_1*instance_num_2, cross_term(i,j) =
%     \sum_{k=1}^{feature_num}w_{k}^{2}*data_matrix_1(i,k)*data_matrix_2(j,k)

distance_matrix = weighted_square_1 + weighted_square_2' - 2 * cross_term;
distance_matrix = max(distance_matrix, 0);
% (1) weighted_square_1 + weighted_square_2 (instance_num_1*1 + 1*instance_num_2 -> instance_num_1*instance_num_2)
%     distance_matrix 是 instance_num_1*instance_num_2 的矩阵,
%     distance_matrix(i,j) 是 data_matrix_1 的第 i 个样本和
%     data_matrix_2 的第 j 个样本的的特征加权距离
% (2) 设置距离大于等于 0, 避免浮点数误差
end