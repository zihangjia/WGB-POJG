function label_pred = dpc_wgb(data_matrix, cluster_num, wgb_num, options)

arguments
    data_matrix (:,:) double {mustBeNonempty, mustBeFinite}
    cluster_num (1,1) double {mustBeInteger, mustBePositive}
    wgb_num (1,1) double {mustBeInteger, mustBePositive} = fix(sqrt(size(data_matrix,1))) + 1
    options (1,1) wgbOptions = wgbOptions()
end

instance_num = size(data_matrix,1);

%% (1) 生成投影粒球
outputs = generate_wgbs(data_matrix, wgb_num, options);

center_matrix = gather(outputs.center_matrix);
feature_weight = gather(outputs.feature_weight);
membership_matrix = gather(outputs.membership_matrix);

clear outputs

if wgb_num <= cluster_num
    [~, label_pred] = max(membership_matrix, [], 2);
    return
end

%% (2) 计算粒球平均半径

distance_matrix = weighted_squared_euclidean_distance( ...
                    data_matrix, center_matrix, feature_weight);
distance_matrix = sqrt(distance_matrix);

weighted_distance_matrix = membership_matrix .* distance_matrix;

membership_sum = sum(membership_matrix,1);
membership_sum(membership_sum==0) = eps;

wgbs_radius_ave = sum(weighted_distance_matrix,1) ./ membership_sum;

%% (3) 计算粒球密度

wgbs_density = membership_sum ./ (wgbs_radius_ave + 1);
wgbs_density = wgbs_density(:);

[~, wgbs_density_index] = sort(wgbs_density,'descend');

%% (4) 粒球中心之间距离

distance_matrix_wgbs = weighted_squared_euclidean_distance( ...
                        center_matrix, center_matrix, feature_weight);
distance_matrix_wgbs = sqrt(distance_matrix_wgbs);

%% (5) 计算相对距离 δ

wgbs_relative_distance = zeros(wgb_num,1);
wgbs_nearest_neighbor = zeros(wgb_num,1);

for k = 2:wgb_num
    
    idx = wgbs_density_index(k);
    
    higher = wgbs_density_index(1:k-1);
    
    [wgbs_relative_distance(idx),pos] = ...
        min(distance_matrix_wgbs(idx,higher));
    
    wgbs_nearest_neighbor(idx) = higher(pos);
    
end

top_idx = wgbs_density_index(1);

wgbs_relative_distance(top_idx) = ...
        max(distance_matrix_wgbs(top_idx,:));

wgbs_nearest_neighbor(top_idx) = top_idx;

%% (6) 决策值 γ

decision_value = wgbs_density .* wgbs_relative_distance;

if all(decision_value < eps)
    label_pred = ones(instance_num,1);
    return
end

[~, decision_index] = sort(decision_value,'descend');

%% (7) 粒球标签传播

wgbs_label_pred = zeros(wgb_num,1);

for k = 1:cluster_num
    wgbs_label_pred(decision_index(k)) = k;
end

for k = 1:wgb_num
    
    idx = wgbs_density_index(k);
    
    if wgbs_label_pred(idx)==0
        
        wgbs_label_pred(idx) = ...
            wgbs_label_pred(wgbs_nearest_neighbor(idx));
        
    end
    
end

%% (8) 映射回样本

[~, idx] = max(membership_matrix,[],2);

label_pred = wgbs_label_pred(idx);

end