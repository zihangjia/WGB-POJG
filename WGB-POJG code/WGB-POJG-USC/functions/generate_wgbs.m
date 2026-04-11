function outputs = generate_wgbs(data_matrix, wgb_num, options)
% 基于合理粒度准则生成投影粒球 generate weighted granular-balls
% 输入:
%   data_matrix - 数据矩阵 (instance_num, feature_num) (非空)
%   wgb_num - 加权粒球的数量 (1,1) (正整数, 默认=sqrt(instance_num))
%   options - 算法的运行参数
% 输出:
%   objective_value - 目标函数向量 (iteration_num+1, 1)
%   center_matrix - 球心矩阵 (wgb_num, feature_num)
%   membership_matrix - 隶属度矩阵 (instance_num, wgb_num)
%   weight_vector - 权重向量 (feature_num, 1)
%   auxiliary_vector_vector - 辅助变量向量 (wgb_num, 1)

% (0) 参数检验 *************************************************************
arguments
    data_matrix (:,:) double {mustBeNonempty}
    wgb_num (1,1) double {mustBeInteger, mustBePositive} = fix(sqrt(size(data_matrix, 1))) + 1
    options (1,1) wgbOptions = wgbOptions()
end
% (0) 参数检验 *************************************************************

% (1) 变量初始化 ***********************************************************
[variables, options] = initialize(data_matrix, wgb_num, options); clear data_matrix wgb_num;
% (1) 变量初始化 ***********************************************************


% (2) 变量优化: 分块坐标下降法 *************************************************************
while variables.iteration_num <= options.max_iteration_num && ~variables.is_convergence

    % (1) 更新辅助变量
    variables.auxiliary_vector = update_auxiliary_vector(variables, options);

    % (2) 更新隶属度矩阵
    variables.membership_matrix = update_membership_matrix(variables, options);

    % (3) 更新球心矩阵
    variables.center_matrix = update_center_matrix(variables, options);

    % (4) 更新权重向量
    variables.feature_weight = update_feature_weight(variables, options);

    % (5) 计算样本与球心之间的加权欧氏距离
    variables.distance_matrix = weighted_squared_euclidean_distance(variables.data_matrix, variables.center_matrix, variables.feature_weight);


    % (6) 计算当前目标函数值, 相对误差和绝对误差
    variables.objective_value(variables.iteration_num + 1) = calculate_objective_value(variables, options, 'all');
    absolute_difference = abs(variables.objective_value(variables.iteration_num + 1) - variables.objective_value(variables.iteration_num));
    relative_difference = absolute_difference / max(eps, abs(variables.objective_value(variables.iteration_num)));
    % —— variables.objective_value(variables.iteration_num + 1) - (1, 1) 第 variables.iteration_num + 1 轮迭代开始时目标函数的值
    % —— absolute_difference - (1, 1) 目标函数经过一轮迭代后的绝对差
    % —— relative_difference - (1, 1) 目标函数经过一轮迭代后的相对差

    % (6) 判断是否收敛
    if absolute_difference < options.convergence_tolerance || relative_difference < options.convergence_tolerance * 1e1
        variables.is_convergence = true;
        break
    else
        variables.iteration_num = variables.iteration_num + 1;
    end
    % 相对误差的收敛容忍度比绝对误差的收敛容忍度大一个数量级
    % 如果目标函数的绝对差或者相对差小于收敛容忍度, 则标记已收敛, 退出循环
    % 如果目标函数的绝对差和相对差都大于收敛容忍度, 则标记未收敛, 迭代次数+1
end
%***************************** 算法迭代 *****************************


% **************************** 算法输出 ****************************
outputs = struct('objective_value', variables.objective_value(1:variables.iteration_num), ...
    'center_matrix', variables.center_matrix, ...
    'membership_matrix', variables.membership_matrix, ...
    'feature_weight', variables.feature_weight, ...
    'auxiliary_vector', variables.auxiliary_vector);
% **************************** 算法输出 ****************************
end

%% 函数区 
function feature_weight = update_feature_weight(variables, options)
%update_feature_weight 更新权重向量
% 检查通过: 2026-03-18 Z. Jia

% 初始化特征权重向量, 目标函数值, 梯度和学习率
feature_weight = variables.feature_weight;
cost_old = compute_cost(feature_weight, variables, options);
gradient = compute_gradient(feature_weight, variables, options);
learning_rate = options.learning_rate_feature_weight / sqrt(variables.iteration_num);
% feature_weight - (feature_num, 1) 初始化特征权重向量为当前迭代的特征权重向量
% cost_old - (1, 1) 计算当前特征权重所对应的目标函数值
% gradient - (feature_num, 1) 计算当前特征权重所对应的梯度
% learning_rate - (1, 1) 初始化学习率为 初始学习率/sqrt(迭代次数)

feature_weight_new = feature_weight .* exp(-learning_rate * gradient);
feature_weight_new = variables.feature_num * feature_weight_new / sum(feature_weight_new);
cost_new = compute_cost(feature_weight_new, variables, options);
% —— feature_weight_new - (feature_num, 1) 采用镜像梯度下降法更新特征权重向量
% —— cost_new - (1, 1) 计算更新后的特征权重向量对应的目标函数值

if cost_new < cost_old
    feature_weight = feature_weight_new;
end
% —— 如果新的目标函数值小于旧的目标函数值, 则更新特征权重向量; 否则, 不更新特征权重向量. 

    function cost = compute_cost(feature_weight, variables, options)
        % 计算关于特征权重向量子问题的目标函数

        distance_matrix = weighted_squared_euclidean_distance(variables.data_matrix, variables.center_matrix, feature_weight);
        membership_distance_matrix = variables.membership_matrix .* distance_matrix;
        difference_matrix = membership_distance_matrix - variables.auxiliary_vector';
        difference_matrix = max(difference_matrix, 0);
        % —— membership_distance_matrix - (instance_num, wgb_num) 样本和球心的加权平方欧氏距离
        % —— difference_matrix - (instance_num, wgb_num) 样本-加权粒球约束违反程度
        % 将约束违反程度为负的值置为 0

        cost_term_1 = sum(membership_distance_matrix, 'all');
        cost_term_2 = options.hyperparameter_penalty * sum(difference_matrix, 'all');
        cost = cost_term_1 + cost_term_2;
        % cost_term_1: 加权平方欧氏距离之和; cost_term_2: 约束违反程度之和
    end

    function gradient = compute_gradient(feature_weight, variables, options)
        % 计算关于特征权重向量子问题的梯度

        distance_matrix = weighted_squared_euclidean_distance(variables.data_matrix, variables.center_matrix, feature_weight);
        membership_distance_matrix = variables.membership_matrix .* distance_matrix;
        difference_matrix = membership_distance_matrix - variables.auxiliary_vector';
        % —— 计算样本与球心之间的加权平方欧氏距离
        % —— 计算隶属度重加权的加权平方欧氏距离
        % —— 计算样本-粒球对违反约束的值

        penalty_index = difference_matrix > 0;
        weight_matrix_with_penalty = variables.membership_matrix .* (1 + options.hyperparameter_penalty * penalty_index);
        % —— 计算样本-粒球违反约束的索引
        % —— 用于梯度计算的系数矩阵

        gradient_term_1 = (variables.data_matrix .* variables.data_matrix)' * sum(weight_matrix_with_penalty, 2); % X' .* X' * (E1)
        gradient_term_2 = (variables.center_matrix .* variables.center_matrix)' * (sum(weight_matrix_with_penalty, 1)'); % C' .* C' * (E'1)
        gradient_term_3 = 2 * sum((variables.data_matrix' * weight_matrix_with_penalty) .* variables.center_matrix', 2); % 2 * diag(X'EC), 所采用的计算方法可以避免构建全部的矩阵
        gradient = 2 * (gradient_term_1 + gradient_term_2 - gradient_term_3) .* feature_weight; % 2 * w .* ((X' .* X' * (E1)) + (C' .* C' * (E'1)) - 2 * diag(X'EC))
    end
end


function center_matrix = update_center_matrix(variables, options)
% 更新球心矩阵
% 检查通过: 2026-03-18 Z. Jia
% 输入:
% —— variables - (1, 1) 算法运行变量
% —— options - (1, 1) 算法运行参数
% 输出:
% —— center_matrix - (wgb_num, feature_num) 球心矩阵

% 初始化回退法参数
linear_search_parameter = struct(...
    'learning_rate', 1,...                      % learning_rate = 1, 初始学习率 
    'reduction_ratio', 0.5,...                  % reduction_ratio = 0.5, 学习率衰减比例
    'max_search_num', 20,...                    % max_search_num = 20, 线搜索最大搜索次数
    'sufficient_descent_coefficient', 1e-4);    % sufficient_descent_coefficient = 1e-4, 充分下降系数
% 线搜索参数

% 初始化球心矩阵, 当前目标函数值和当前梯度
center_matrix = variables.center_matrix;
cost_old = compute_cost(center_matrix, variables, options);
gradient = compute_gradient(center_matrix, variables, options);
is_sufficient_descent = false;
% center_matrix - (wgb_num, feature_num) 初始化球心矩阵为当前迭代的球心矩阵
% cost_old - (1, 1) 计算当前球心矩阵所对应的目标函数值
% gradient - (wgb_num, feature_num) 计算当前球心矩阵所对应的梯度
% is_descent - (1, 1) 新的球心矩阵是否使得目标函数下降

for search_num = 1:linear_search_parameter.max_search_num

    center_matrix_new = center_matrix - linear_search_parameter.learning_rate * gradient;
    cost_new = compute_cost(center_matrix_new, variables, options);
    % —— feature_weight_new - (feature_num, 1) 基于当前学习率计算新特征权重向量
    % —— cost_new - (1, 1) 新的特征权重向量所对应的目标函数

    if cost_new <= cost_old - linear_search_parameter.sufficient_descent_coefficient * linear_search_parameter.learning_rate * sum(gradient .* gradient, 'all')
    % f(x-ag) <= f(x) - c*a*||g||_{2}^{2}, a 是学习率, c 是充分下降系数, g 是梯度
        is_sufficient_descent = true;
        break;
    else
        linear_search_parameter.learning_rate = linear_search_parameter.reduction_ratio * linear_search_parameter.learning_rate;
    end
    % —— case 1: 目标函数下降: 设置 is_descent 为 true 并退出循环
    % —— case 2: 目标函数未下降: 更新学习率
end
% —— 最多进行 max_search_num 次学习率搜索

if is_sufficient_descent
    center_matrix = center_matrix_new;
end
% 如果目标函数下降, 则更新球心矩阵; 否则, 球心矩阵为初始值

    function cost = compute_cost(center_matrix, variables, options)
        % 计算关于球心矩阵的子问题的目标函数

        distance_matrix = weighted_squared_euclidean_distance(variables.data_matrix, center_matrix, variables.feature_weight);
        membership_distance_matrix = variables.membership_matrix .* distance_matrix;
        difference_matrix = membership_distance_matrix - variables.auxiliary_vector';
        difference_matrix = max(difference_matrix, 0);
        % —— distance_matrix - (instance_num, wgb_num) 样本和当前球心矩阵的加权平方欧氏距离矩阵
        % —— membership_distance_matrix - (instance_num, wgb_num) 样本和球心的隶属度重加权的加权平方欧氏距离矩阵
        % —— difference_matrix - (instance_num, wgb_num) 样本-球心-辅助变量约束违反程度
        % 将约束违反程度为负的值置为 0

        cost_term_1 = sum(membership_distance_matrix, 'all');
        cost_term_2 = options.hyperparameter_penalty * sum(difference_matrix, 'all');
        cost = cost_term_1 + cost_term_2;
        % cost_term_1: 隶属度重加权的加权平方欧氏距离之和; cost_term_2: 约束违反程度之和
    end

    function gradient = compute_gradient(center_matrix, variables, options)
        % 计算关于球心矩阵的子问题的梯度

        distance_matrix = weighted_squared_euclidean_distance(variables.data_matrix, center_matrix, variables.feature_weight);
        membership_distance_matrix = variables.membership_matrix .* distance_matrix;
        difference_matrix = membership_distance_matrix - variables.auxiliary_vector';
        % —— distance_matrix - (instance_num, wgb_num) 样本和当前球心矩阵的加权平方欧氏距离矩阵
        % —— membership_distance_matrix - (instance_num, wgb_num) 样本和球心的隶属度重加权的加权平方欧氏距离矩阵
        % —— difference_matrix - (instance_num, wgb_num) 样本-球心-辅助变量约束违反程度

        penalty_index = difference_matrix > 0;
        coefficient_matrix = variables.membership_matrix .* (1 + options.hyperparameter_penalty * penalty_index);
        % —— penalty_index - (instance_num, wgb_num) 违反约束的样本-粒球对的逻辑矩阵
        % —— coefficient_matrix - (instance_num, wgb_num) 用于计算梯度的系数矩阵

        gradient_term_1 = sum(coefficient_matrix, 1)' .* center_matrix; % diag(E'1)C
        gradient_term_2 = coefficient_matrix' * variables.data_matrix; % E'X
        gradient_term_3 = (variables.feature_weight .^ 2)'; % diag(w \odot w)
        gradient = 2 * (gradient_term_1 - gradient_term_2) .* gradient_term_3; % 2 (diag(E'1)C - E'X) diag(w \odot w)
    end
end


function membership_matrix = update_membership_matrix(variables, options)
% 更新隶属度矩阵 检查通过: 2026-03-18 Z. Jia
% 输入:
% —— variables - (1, 1) 算法运行变量
% —— options - (1, 1) 算法运行参数
% 输出:
% —— membership_matrix - (instance_num, wgb_num) 隶属度矩阵

wgb_quality_term = -1 * options.hyperparameter_pojg ./ (1 + options.hyperparameter_granularity * variables.auxiliary_vector');
constraint_penalty_term = options.hyperparameter_penalty * (variables.distance_matrix - variables.auxiliary_vector');
constraint_penalty_term = max(constraint_penalty_term, 0);
% wgb_quality - (1, wgb_num) 每个投影粒球的质量
% constraint_penalty - (instance_num, wgb_num) 样本关于粒球违反约束的程度
% constraint_penalty 中所有小于 0 的项设置为 0

correct_distance = variables.distance_matrix + wgb_quality_term + constraint_penalty_term;
[~, min_index] = min(correct_distance, [], 2);
membership_matrix = gpuArray.false(variables.instance_num, variables.wgb_num);
membership_matrix(sub2ind([variables.instance_num, variables.wgb_num], (1:variables.instance_num)', min_index)) = true;
% correct_distance - (instance_num, wgb_num) 用于分配样本关于投影粒球隶属度的距离
% min_index - (instance_num, 1) 每个样本应该被分配到哪个投影粒球的索引
% membership_matrix - (instance_num, wgb_num) 隶属度矩阵
end


function auxiliary_vector = update_auxiliary_vector(variables, options)
% 更新辅助向量, 理论上时间复杂度小于 O(wgb_num * instance_num * log(instance_num))
% 检查通过: 2026-03-18 Z. Jia
% 输入:
% —— variables - (1,1) 算法当前所使用的变量
% —— options - (1,1) 算法运行参数
% 输出:
% —— auxiliary_vector - (wgb_num, 1) 辅助变量

% (1) 构造辅助变量解集 *****************************************************
membership_distance_matrix = variables.membership_matrix .* variables.distance_matrix;
membership_distance_matrix = sort(membership_distance_matrix, 1, "ascend");
membership_distance_matrix = [gpuArray.zeros(1, variables.wgb_num); membership_distance_matrix];
% —— membership_distance_matrix - (instance_num, wgb_num) 隶属度矩阵与加权欧氏距离矩阵的逐元素积
% —— 将 membership_distance_matrix 矩阵按列升序排序
% —— membership_distance_matrix - (instance_num+1, wgb_num) 将 0 纳入每个辅助变量的解集
% (1) 构造辅助变量解集 *****************************************************

% (2) 计算每个候选解对应的目标函数值 *****************************************
membership_distance_matrix_sum = sum(membership_distance_matrix, 1) - cumsum(membership_distance_matrix, 1);
membership_matrix_sum = sum(variables.membership_matrix, 1);
inverse_index = (variables.instance_num:-1:0)';
first_term = -1 * options.hyperparameter_pojg * membership_matrix_sum ./ (1 + options.hyperparameter_granularity * membership_distance_matrix);
second_term = options.hyperparameter_penalty * (membership_distance_matrix_sum - inverse_index .* membership_distance_matrix);
objective_value = first_term + second_term;
% —— weighted_distance_matrix_sum - (instance_num+1, wgb_num): 第 (i,j) 个元素
%    表示关于第 j 个加权粒球的第 i+1, i+2,..., instance_num 大的加权平方欧氏距离和
% —— membership_matrix_sum - (1, wgb_num): 第 (1,j) 个元素表示属于第 j 个投影粒球的样本数量
% —— inverse_index - (instance_num+1, 1) 生成逆序数列向量
% —— 目标函数第 1 项: -\lambda_{1} * \frac{\sum_{i=1}^{n}a_{i,j}}{1+\gamma h_{j}}
% —— 目标函数第 2 项: \sigma * (\sum_{k=q+1}^{n}wd_{i,j} - (n-q)wd_{i,j})
% (2) 计算每个候选解对应的目标函数值 *****************************************

% (3) 计算每个辅助变量的值 **************************************************
[~, auxiliary_vector_index] = min(objective_value, [], 1);
linear_indices = sub2ind([variables.instance_num+1, variables.wgb_num], auxiliary_vector_index, 1:variables.wgb_num);
auxiliary_vector = membership_distance_matrix(linear_indices)';
% —— auxiliary_vector_index - (1, wgb_num): 辅助变量在解集 weighted_distance_matrix 中的行索引
% —— linear_indices - (1, wgb_num): 辅助变量在解集 weighted_distance_matrix 中的线性索引
% —— auxiliary_vector - (wgb_num, 1): 辅助变量列向量
% (3) 计算每个辅助变量的值 **************************************************
end


function [objective_value, objective_value_term] = calculate_objective_value(variables, options, type)
% 计算目标函数值
% 检查通过: 2026-03-18 Z. Jia
% 输入:
% —— variables - (1,1) 算法当前所使用的变量
% —— options - (1,1) 算法运行参数
% 输出:
% —— objective_value - (1,1) 目标函数值

if nargin < 3
    type = 'all';
end

switch type
    case 'all'
        membership_distance_matrix = variables.membership_matrix .* variables.distance_matrix;
        first_term = sum(membership_distance_matrix, 'all');
        % weighted_distance_matrix - (instance_num, wgb_num) 隶属度矩阵和距离矩阵的逐元素乘积
        % first_term - (1, 1) 目标函数第 1 项, 低维空间中样本到球心的加权平方欧氏距离和

        second_term = -1 * options.hyperparameter_pojg * sum(variables.membership_matrix ./ (1 + options.hyperparameter_granularity * variables.auxiliary_vector'), 'all');
        % third_term - (1, 1) 目标函数第 2 项, 加权粒球的质量和

        difference_matrix = membership_distance_matrix - variables.auxiliary_vector';
        difference_matrix = max(difference_matrix, 0);
        third_term = options.hyperparameter_penalty * sum(difference_matrix, 'all');
        % difference_matrix - (instance_num, wgb_num) 精确惩罚项约束的违反程度
        % difference_matrix 违反程度小于 0 的项等于未违反约束
        % fourth_term - (1, 1) 目标函数第 3 项, 精确惩罚项

        objective_value = first_term + second_term + third_term;
        objective_value_term = [first_term; second_term; third_term];
        % objective_value - (1, 1) 目标函数值
        % objective_value_term - (3, 1) 目标函数各个项的值
end
end

 
function distance_matrix = weighted_squared_euclidean_distance(data_matrix_1, data_matrix_2, feature_weight)
%weighted_squared_euclidean_vector 输入 2 个样本观测矩阵, 返回样本之间的距离矩阵
% 检查通过: 2026-03-18 Z. Jia
% 输入:
%   data_matrix_1 - (instance_num_1, feature_num)
%   data_matrix_2 - (instance_num_2, feature_num)
%   feature_weight - (feature_num, 1)
% 输出:
%   distance_matrix - (instance_num_1, instance_num_2)

feature_weight = feature_weight(:);
% 确保 feature_weight 是列向量

weighted_data_1 = data_matrix_1 .* feature_weight';
weighted_data_2 = data_matrix_2 .* feature_weight';
% weighted_data_1(i, k) = data_matrix_1(i, k) * w(k)
% weighted_data_2(j, k) = data_matrix_2(j, k) * w(k)

weighted_square_1 = sum(weighted_data_1 .^ 2, 2);
weighted_square_2 = sum(weighted_data_2 .^ 2, 2);
% weighted_square_1(i) = sum_{k=1}^{M} data_matrix_1(i, k)^2 * w(k)^2
% weighted_square_2(j) = sum_{k=1}^{M} data_matrix_2(j, k)^2 * w(k)^2

cross_term = weighted_data_1 * weighted_data_2';
% cross_term(i, j) = sum_{k=1}^{M} w(k)^2 * data_matrix_1(i, k) * data_matrix(j, k)

distance_matrix = weighted_square_1 + weighted_square_2' - 2 * cross_term;
distance_matrix = max(distance_matrix, 0);
% distance_matrix(i, j) = 
%   sum_{k=1}^{M} data_matrix_1(i, k)^2 * w(k)^2 
% + sum_{k=1}^{M} data_matrix_2(j, k)^2 * w(k)^2 
% - 2*sum_{k=1}^{M} w(k)^2 * data_matrix_1(i, k) * data_matrix(j, k) 
% 设置距离大于等于 0, 避免浮点数误差
end


function [center_matrix, distance_matrix] = initialize_center_matrix(data_matrix, wgb_num, feature_weight)
%initialize_center_matrix 通过与 k-means++ 一样的方式初始化球心矩阵
% 检查通过: 2026-03-18 Z. Jia
% 输入:
%   data_matrix - (instance_num, feature_num) 数据矩阵
%   wgb_num - (1, 1) 加权粒球数量
%   feature_weight - (feature_num, 1) 特征权重向量
% 输出:
%   center_matrix - (wgb_num, feature_num) 球心矩阵
%   distance_matrix - (instance_num, wgb_num) 样本到球心的加权平方欧氏距离

% (1) 初始化 **************************************************************************************************************************
[instance_num, feature_num] = size(data_matrix);
if nargin < 3
    feature_weight = ones(feature_num, 1);
end
% —— instance_num, feature_num - (1, 1) 样本和特征数量
% —— feature_weight - (feature_num, 1) 如果没有输入特征权重, 初始化特征权重向量为全 1 向量

center_matrix = nan(wgb_num, feature_num);
index = randi(instance_num);
center_matrix(1, :) = data_matrix(index, :);
center_selected_num = 1;
% —— center_matrix - (wgb_num, feature_num) 初始化球心矩阵为全 nan 矩阵
% —— index - (1, 1) 从 1 到 instance_num 之间随机选取一个整数
% —— 将第 1 个球心设置为第 index 个样本
% —— center_selected_num - (1, 1) 已选择的球心的数量

distance_matrix = nan(instance_num, wgb_num);
distance_matrix(:, 1) = weighted_squared_euclidean_distance(data_matrix, center_matrix(1, :), feature_weight);
% —— distance_matrix - (instance_num, wgb_num) 初始化距离矩阵
% —— distance_matrix 第 1 列为样本到第 1 个球心的加权平方欧氏距离
% (1) 初始化 **************************************************************************************************************************

while center_selected_num < wgb_num
    distance_min = min(distance_matrix(:, 1:center_selected_num), [], 2);
    % —— distance_min - (instance_num, 1) 样本到已选中球心的最小距离

    if sum(distance_min) < eps
        center_matrix(center_selected_num + 1, :) = data_matrix(randi(instance_num), :); 
    else
        center_matrix(center_selected_num + 1, :) = datasample(data_matrix, 1, 1, 'Weights', distance_min, 'Replace', false);
    end
    % —— 如果所有样本到球心的最小距离都等于 0, 那么随机挑选一个球心
    % —— 否则通过 distance_min 对应的概率分布随机确定一个球心

    distance_matrix(:, center_selected_num + 1) = weighted_squared_euclidean_distance(data_matrix, center_matrix(center_selected_num + 1, :), feature_weight);
    % —— 计算样本到新球心的加权平方欧氏距离

    center_selected_num = center_selected_num + 1;
    % —— 已选中的粒球球心数量 + 1
end
% —— 当已选中的球心的数量 center_selected_num 小于加权粒球的数量 wgb_num 时循环
end


function [variables, options] = initialize(data_matrix, wgb_num, options)
% initialize 初始化算法
% 检查通过: 2026-03-18 Z. Jia

% (1) 初始化决策变量和距离矩阵 *********************************************************************************************************
[instance_num, feature_num] = size(data_matrix);
% —— instance_num, feature_num - (1, 1) 样本和特征的数量

feature_weight = ones(feature_num, 1);
% —— feature_weight - (feature_num, 1) 初始化特征权重向量为全 1 向量

[center_matrix, distance_matrix] = initialize_center_matrix(data_matrix, wgb_num, feature_weight);
% —— center_matrix - (wgb_num, feature_num) 采用与 k-means++ 相同的方式初始化球心矩阵
% —— distance_matrix - (instance_num, wgb_num) 样本与球心的加权平方欧氏距离矩阵

[~, label_pred] = min(distance_matrix, [], 2);
membership_matrix = false(instance_num, wgb_num);
membership_matrix(sub2ind([instance_num, wgb_num], (1:instance_num)', label_pred)) = true;
% —— label_pred - (instance_num, 1) 样本的标签向量
% —— membership_matrix - (instance_num, wgb_num) 初始化隶属度矩阵
% —— 根据样本的标签进一步初始化隶属度矩阵

auxiliary_vector = max(distance_matrix .* membership_matrix, [], 1)';
% —— auxiliary_vector - (wgb_num, 1) 初始化辅助变量为样本到球心的最大平方欧氏距离
% (1) 初始化决策变量和距离矩阵 *********************************************************************************************************


% (2) 将变量存储至结构体 variables 中 **************************************************************************************************
variables.data_matrix = gpuArray(data_matrix);                      % 数据矩阵
variables.instance_num = instance_num;                              % 样本数量
variables.feature_num = feature_num;                                % 特征数量
variables.wgb_num = wgb_num;                                        % 加权粒球数量
% 常量

variables.auxiliary_vector = gpuArray(auxiliary_vector);            % 辅助变量
variables.membership_matrix = gpuArray(membership_matrix);          % 隶属度矩阵
variables.center_matrix = gpuArray(center_matrix);                  % 球心矩阵
variables.feature_weight = gpuArray(feature_weight);                % 特征权重向量
variables.distance_matrix = gpuArray(distance_matrix);              % 距离矩阵
% 变量
% (2) 将变量存储至结构体 variables 中 **************************************************************************************************


% (3) 初始化算法迭代所需变量 ********************************************
variables.is_convergence = false;
variables.iteration_num = 1;
variables.objective_value = nan(options.max_iteration_num + 1, 1);
% —— variables.is_convergence - (1,1), 算法是否收敛
% —— variables.iteration_num - (1,1), 算法当前迭代次数
% —— variables.objective_value - (max_iteration_num+1, 1) 算法第 i 次迭代开始之前的目标函数值

[~, objective_value_term] = calculate_objective_value(variables, options, 'all'); objective_value_term = gather(objective_value_term);
options.hyperparameter_pojg = options.hyperparameter_pojg * abs(objective_value_term(1)) / max(abs(objective_value_term(2)), eps);
variables.objective_value(1) = gather(calculate_objective_value(variables, options, 'all'));
% —— objective_value_term - (3, 1) 初始化所有变量后目标函数各项的值
% —— 调整加权粒球质量项的超参数, 使得加权粒球相似项和加权粒球质量项有相同的数量级: parameter_i = parameter_i * abs(term_1)/ max(abs(term_i), eps)
% —— 计算目标函数在第 1 次迭代开始之前的目标函数值
% (3) 初始化算法迭代所需变量 ********************************************
end