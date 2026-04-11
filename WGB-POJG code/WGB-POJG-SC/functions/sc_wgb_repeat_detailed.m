function outputs = sc_wgb_repeat_detailed(data_matrix, label, cluster_num, wgb_num, options, repeat_num, metric_name)
%SC_WGB_REPEAT  多次运行 SC-WGB 聚类算法并返回平均性能指标
%
%   metric_ave = SC_WGB_REPEAT(data_matrix, label, cluster_num, wgb_num, options, repeat_num, metric_name)
%
%   输入参数:
%   ------------------------------------------------------------
%   data_matrix : n × d 数据矩阵
%   label       : 真实类别标签 (用于评价指标计算)
%   cluster_num : 聚类类别数
%   wgb_num     : weighted granular-ball 数目
%   options     : WGB 算法的参数结构体 (由 wgbOptions 生成)
%   repeat_num  : 重复运行次数 (默认 20)
%   metric_name : 性能指标名称:
%                 'nmi' —— Normalized Mutual Information
%                 'ari' —— Adjusted Rand Index
%                 'purity' —— purity
%
%   输出参数:
%   ------------------------------------------------------------
%   metric_ave  : 指标的负平均值 (用于 bayesopt 最小化目标函数)
%
%   说明:
%   ------------------------------------------------------------
%   (1) 每次调用本函数时固定随机种子 rng(1)，
%       使不同参数组合在相同随机初始化条件下比较。
%   (2) 内部重复运行 repeat_num 次，取平均指标。
%   (3) 最终返回 -mean(metric)，
%       以适配 MATLAB bayesopt 默认的“最小化”设定。
%

% ------------------ 参数默认值 ------------------------

if nargin < 6 || isempty(repeat_num)
    repeat_num = 20;
end

if nargin < 7 || isempty(metric_name)
    metric_name = 'nmi';
end

% ------------------ 初始化存储 ------------------------
if strcmp(metric_name, 'all')
    metric = zeros(repeat_num, 3);
else
    metric = zeros(repeat_num, 1);
end

% 固定随机种子，保证每次目标函数评估具有可比性
rng(1);

% ------------------ 重复运行算法 ----------------------
time = 0;

for item = 1:repeat_num

    tic
    label_pred = sc_wgb(data_matrix, cluster_num, wgb_num, options);
    time = time + toc;

    switch lower(metric_name)

        case 'nmi'
            metric(item) =  clustering_metric.compute_nmi(label, label_pred) * 100;

        case 'ari'
            metric(item) = clustering_metric.compute_ari(label, label_pred) * 100;

        case 'purity'
            metric(item) = clustering_metric.compute_purity(label, label_pred) * 100;

        case 'all'
            metric(item, 1) = clustering_metric.compute_nmi(label, label_pred) * 100;
            metric(item, 2) = clustering_metric.compute_ari(label, label_pred) * 100;
            metric(item, 3) = clustering_metric.compute_purity(label, label_pred) * 100;

        otherwise
            error('未指定该度量指标: %s', metric_name);
    end
end

% ------------------ 返回 bayesopt 目标 ----------------

% bayesopt 默认最小化，因此返回负均值
[metric_std, metric_ave] = std(metric);
metric_ave = metric_ave * -1;
time_ave = time / repeat_num;

outputs = struct('metric_std', metric_std, 'metric_ave', metric_ave, 'time_ave', time_ave);
end
