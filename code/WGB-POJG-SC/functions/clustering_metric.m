classdef clustering_metric
    %METRIC metric used to evalute the performance of clustering results

    methods(Static)
        function [AR,RI,MI,HI] = compute_ari(label, label_pred)
            % 计算Rand指数及其变体
            % 输入：c1, c2 - 聚类标签向量
            % 输出：AR - Adjusted Rand Index
            %       RI - Rand Index
            %       MI - Mirkin Index
            %       HI - Hubert Index

            % 输入验证
            if nargin < 2
                error('valid_RandIndex: 需要两个输入参数');
            end

            if ~isvector(label) || ~isvector(label_pred)
                error('valid_RandIndex: 输入必须是向量');
            end

            label = label(:);
            label_pred = label_pred(:);

            if length(label) ~= length(label_pred)
                error('valid_RandIndex: 两个输入向量长度必须相同');
            end

            if length(label) < 2
                error('valid_RandIndex: 至少需要2个样本');
            end

            % 转换标签为连续整数（从1开始）
            [~, ~, ic1] = unique(label);
            [~, ~, ic2] = unique(label_pred);

            % 构建列联表
            C = accumarray([ic1, ic2], 1);

            n = sum(C(:));  % 总样本数

            % 计算行和列的和
            row_sums = sum(C, 2);
            col_sums = sum(C, 1);

            % 计算总对数
            total_pairs = n*(n-1)/2;

            % 计算各类对的数量
            % TP: 同在一类（在两个聚类中）
            TP = sum(sum(C.*(C-1)/2));

            % 计算行组合数和列组合数
            row_pairs = sum(row_sums.*(row_sums-1)/2);
            col_pairs = sum(col_sums.*(col_sums-1)/2);

            % 计算其他类型的对
            FP = row_pairs - TP;      % 在c1中同簇，在c2中不同簇
            FN = col_pairs - TP;      % 在c2中同簇，在c1中不同簇
            TN = total_pairs - TP - FP - FN;  % 在两个聚类中都不同簇

            % 计算各种指数
            RI = (TP + TN) / total_pairs;          % Rand指数
            MI = (FP + FN) / total_pairs;          % Mirkin指数（标准化版本）
            HI = (TP - (FP + FN)) / total_pairs;   % Hubert指数

            % 计算调整的Rand指数
            expected_TP = row_pairs * col_pairs / total_pairs;
            max_TP = 0.5 * (row_pairs + col_pairs);

            if max_TP == expected_TP
                AR = 0;  % 避免除零
            else
                AR = (TP - expected_TP) / (max_TP - expected_TP);
            end

        end
        
        function nmi = compute_nmi(label, label_pred)
            % ============================================================
            % Normalized Mutual Information compatible with
            % sklearn.metrics.normalized_mutual_info_score
            % (default average_method = 'arithmetic')
            %
            % Fully vectorized implementation (no loops)
            % Uses natural logarithm.
            % ============================================================

            % 输入验证
            if nargin < 2
                error('需要两个输入参数');
            end

            label = double(label(:));
            label_pred = double(label_pred(:));

            if length(label) ~= length(label_pred)
                error('标签向量长度必须相同');
            end

            N = numel(label);

            % 如果只有一个样本，直接返回1
            if N <= 1
                nmi = 1;
                return;
            end

            % Relabel to 1..K
            [~,~,label] = unique(label);
            [~,~,label_pred] = unique(label_pred);

            Kt = max(label);
            Kp = max(label_pred);

            % ------------------------------------------------------------
            % Contingency matrix (cluster x class)
            % ------------------------------------------------------------
            C = sparse(label_pred, label, 1, Kp, Kt);

            nij = full(C);

            % 添加一个小常数避免log(0)
            epsilon = 1e-12;

            % Probabilities
            pij = nij / N + epsilon;
            pi  = sum(pij,1);   % true label marginal
            pj  = sum(pij,2);   % cluster marginal

            % ------------------------------------------------------------
            % Mutual Information: sum pij * log(pij/(pi*pj))
            % ------------------------------------------------------------
            % 更稳健的计算方式
            outer = pj * pi;

            % 避免除零和log(0)
            valid_mask = (pij > epsilon) & (outer > epsilon);

            % 计算互信息
            if any(valid_mask(:))
                ratio = pij(valid_mask) ./ outer(valid_mask);
                MI = sum( pij(valid_mask) .* log(ratio) );
            else
                MI = 0;
            end

            % ------------------------------------------------------------
            % Entropies (添加小常数避免log(0))
            % ------------------------------------------------------------
            Ht = -sum( pi .* log(pi + epsilon) );
            Hp = -sum( pj .* log(pj + epsilon) );

            % ------------------------------------------------------------
            % sklearn normalization (arithmetic)
            % ------------------------------------------------------------
            den = (Ht + Hp)/2;

            if den <= epsilon
                nmi = 1;
            else
                nmi = MI / den;
            end

            % 确保结果在[0,1]范围内
            nmi = max(0, min(1, nmi));

        end

        function purity = compute_purity(label, label_pred)
            %CALCULATE_PURITY_VEC  Compute clustering purity efficiently.
            %
            % true_labels : n×1 ground-truth labels
            % pred_labels : n×1 cluster labels

            label = label(:);
            label_pred = label_pred(:);

            % 映射为连续索引
            [~,~,true_idx] = unique(label);
            [~,~,pred_idx] = unique(label_pred);

            % 构造混淆矩阵（行=真实类，列=预测簇）
            confusion_matrix = accumarray( ...
                [true_idx, pred_idx], ...
                1);

            % 每个预测簇中最多的真实类别数
            cluster_max = max(confusion_matrix, [], 1);

            % purity
            purity = sum(cluster_max) / numel(label);
        end
    end
end

