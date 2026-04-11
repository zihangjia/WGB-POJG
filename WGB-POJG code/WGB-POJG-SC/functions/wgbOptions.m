classdef wgbOptions
    %WGBOPTIONS 基于合理粒度准则的加权粒球生成算法的参数配置类
    %   用于配置 generate weighted granular-balls 算法的运行参数
    % 属性:
    %   hyperparameter_pojg             超参数: 加权粒球质量项 (非负数, 默认=1)
    %   hyperparameter_granularity      超参数: 粒度水平 (非负数, 默认=1)
    %   hyperparameter_penalty          超参数: 精确罚函数项 (非负数, 默认=1)
    %   learning_rate_feature_weight    超参数: 特征权重学习率 (正有限数, 默认=1e-1)
    %   max_iteration_num               最大迭代次数 (正整数, 默认=100)
    %   convergence_tolerance           收敛容差 (正有限数, 默认=1e-4)
    %   2026.03.18 checked by Z. Jia

    properties
        hyperparameter_pojg             % 超参数: 加权粒球质量项
        hyperparameter_granularity      % 超参数: 粒度水平
        hyperparameter_penalty          % 超参数: 精确罚函数项
        learning_rate_feature_weight    % 超参数: 特征权重学习率
        max_iteration_num               % 最大迭代次数
        convergence_tolerance           % 收敛容差
    end
    
    methods
        function obj = wgbOptions(args)
            arguments
                args.hyperparameter_pojg (1,1) double {mustBeNonnegative} = 1
                args.hyperparameter_granularity (1,1) double {mustBeNonnegative} = 1
                args.hyperparameter_penalty (1,1) double {mustBeNonnegative} = 1
                args.learning_rate_feature_weight (1,1) double {mustBePositive, mustBeFinite} = 1e-1
                args.max_iteration_num (1,1) double {mustBePositive, mustBeInteger} = 100
                args.convergence_tolerance (1,1) double {mustBePositive, mustBeFinite} = 1e-4
            end
            
            obj.hyperparameter_pojg = args.hyperparameter_pojg;
            obj.hyperparameter_granularity = args.hyperparameter_granularity;
            obj.hyperparameter_penalty = args.hyperparameter_penalty;
            obj.learning_rate_feature_weight = args.learning_rate_feature_weight;
            obj.max_iteration_num = args.max_iteration_num;
            obj.convergence_tolerance = args.convergence_tolerance;
        end
    end
end

