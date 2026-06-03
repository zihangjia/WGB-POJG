clear all

initialize_path();

load("experimental_result.mat");

dataset_num = 18;

results = struct(...
    'nmi_ave', zeros(dataset_num, 1), 'nmi_std', zeros(dataset_num, 1),...
    'ari_ave', zeros(dataset_num, 1), 'ari_std', zeros(dataset_num, 1),...
    'time_ave', zeros(dataset_num, 1));

for no = 1:dataset_num
    fprintf('\n=================================================\n');
    fprintf('正在计算第 %d 个数据集\n', no);
    fprintf('=================================================\n');

    % ------------------ 载入数据 --------------------------
    file_name = ['D', num2str(no), '.mat'];
    load(file_name);
    [instance_num, feature_num] = size(data);
    fprintf('样本数: %d  特征数: %d\n', instance_num, feature_num);
    
    % ------------------ 数据归一化 ------------------------
    % Min-Max 归一化
    % 先缓存 min 和 max 避免重复计算
    data_min = min(data, [], 1); 
    data_max = max(data, [], 1);
    range = data_max - data_min;
    range(range == 0) = 1;
    data = (data - data_min) ./ range;

    % 复现 NMI 的结果
    parameter = performance.nmi_parameter(no, :);
    wgb_num = parameter(2);
    options = wgbOptions("hyperparameter_pojg", parameter(1), "learning_rate_feature_weight", parameter(3));
    outputs = sc_wgb_repeat_detailed(data, label, class_num, wgb_num, options, 20, 'nmi');
    results.nmi_ave(no) = -1 * outputs.metric_ave;
    results.nmi_std(no) = outputs.metric_std;
    

    % 复现 ARI 的结果
    parameter = performance.ari_parameter(no, :);
    wgb_num = parameter(2);
    options = wgbOptions("hyperparameter_pojg", parameter(1), "learning_rate_feature_weight", parameter(3));
    outputs = sc_wgb_repeat_detailed(data, label, class_num, wgb_num, options, 20, 'ari');
    results.ari_ave(no) = -1 * outputs.metric_ave;
    results.ari_std(no) = outputs.metric_std;
    results.time_ave(no) = outputs.time_ave;
    save('reproduce.mat', "results");
end


%%
function [] = initialize_path()
% 初始化函数路径与数据集路径

addpath(fullfile(pwd, 'functions'));
addpath(fullfile(pwd, '..', '..', 'datasets'));

end