% function [S] = L1QP_FeatureSign_Set(X, B, Sigma, beta, gamma)
% 
% [dFea, nSmp] = size(X);
% nBases = size(B, 2);
% 
% % sparse codes of the features
% S = sparse(nBases, nSmp);
% 
% A = double(B'*B + 2*beta*Sigma);
% 
% for ii = 1:nSmp,
%     b = -B'*X(:, ii);
% %     [net] = L1QP_FeatureSign(gamma, A, b);
%     S(:, ii) = L1QP_FeatureSign_yang(gamma, A, b);
% end




function [V] = L1QP_FeatureSign_Set(X, U, Sigma, beta, gamma)

% [dFea, nSmp] = size(X);
% nBases = size(B, 2);
% 
% % sparse codes of the features
% S = sparse(nBases, nSmp);
% 
% A = double(B'*B + 2*beta*Sigma);
% 
% for ii = 1:nSmp,
%     b = -B'*X(:, ii);
% %     [net] = L1QP_FeatureSign(gamma, A, b);
%     S(:, ii) = L1QP_FeatureSign_yang(gamma, A, b);






% 第一小步
[dFea, nSmp] = size(X);
nBases = size(U, 2);
Sigma = 100;
z_label = nBases;
y_label = nSmp; 
K = 5;

%K最近邻关系暂时用欧式距离
tmp_distance = zeros(1,y_label);
W = zeros(y_label,y_label);
for i = 1:y_label
    for j = 1:y_label
        tmp_distance(1,j) = sqrt(sum((X(:,i) - X(:,j)).^2));
%         tmp2 = 0;
%         for r = 1:x_label
%             if(X(r,i) < X(r,j))
%                 tmp2 = tmp2 + X(r,i);
%             else
%                 tmp2 = tmp2 + X(r,j);
%             end
%         end
%         tmp_distance(1,j) = tmp2;   %直方图距离
    end
        [~,sort_num] = sort(tmp_distance);
        W(i,sort_num(1,1:(K+1))) = 1;
        W(i,i) = 0;
end

%计算对角矩阵D
D = zeros(y_label,y_label);
for i = 1:y_label
    D(i,i) = sum(W(i,:));
end
L = D - W;




sgam = 0.1;
lamd = 0.4;
beta = 0.2;



V = sprand(z_label,y_label,0.05);
% V = full(V);
d =zeros(z_label,y_label);
for i = 1:z_label
    for j = 1:y_label
        dist = sqrt(sum((X(:,j) - U(:,i)).^2));
        d(i,j) = exp(dist / Sigma);
    end
end
tmp_eye = eye(z_label,z_label);

tmp_online = U'*X+beta*V*W;
for j = 1:y_label
    tmp_11 =  U'*U*V; 
    tmp_11 = tmp_11 + beta*V*D;
    tmp_a = (d(:,j)).^2;
    for r = 1:z_label
        tmp_eye(r,r) = tmp_a(r,1);
    end
    tmp_b = lamd .* tmp_eye *V;
    tmp_11 = tmp_11 + tmp_b;
    for i = 1:z_label
        V(i,j) = V(i,j) * tmp_online(i,j) / tmp_11(i,j);
    end
end
end