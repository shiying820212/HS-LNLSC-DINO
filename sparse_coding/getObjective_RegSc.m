function [fobj, fresidue, fsparsity, fregs] = getObjective_RegSc(X, B, S, Sigma, beta, gamma)

Err = X - B*S;

fresidue = 0.5*sum(sum(Err.^2));

% alpha=0.1;

% [dFea, nSmp] = size(X);
% nBases = size(B, 2);
% z_label = nBases;
% y_label = nSmp; 
% K = 5;
% 
% %K最近邻关系暂时用欧式距离
% tmp_distance = zeros(1,y_label);
% W = zeros(y_label,y_label);
% for i = 1:y_label
%     for j = 1:y_label
%         tmp_distance(1,j) = sqrt(sum((X(:,i) - X(:,j)).^2));
% %         tmp2 = 0;
% %         for r = 1:x_label
% %             if(X(r,i) < X(r,j))
% %                 tmp2 = tmp2 + X(r,i);
% %             else
% %                 tmp2 = tmp2 + X(r,j);
% %             end
% %         end
% %         tmp_distance(1,j) = tmp2;   %直方图距离
%     end
%         [~,sort_num] = sort(tmp_distance);
%         W(i,sort_num(1,1:(K+1))) = 1;
%         W(i,i) = 0;
% end
% 
% %计算对角矩阵D
% D = zeros(y_label,y_label);
% for i = 1:y_label
%     D(i,i) = sum(W(i,:));
% end
% L = D - W;
% 
% d =zeros(z_label,y_label);
% for i = 1:z_label
%     for j = 1:y_label
%         dist = sqrt(sum((X(:,j) - B(:,i)).^2));
%         d(i,j) = exp(dist / Sigma);
%     end
% end

 fsparsity = gamma*sum(sum(abs(S)));
% fsparsity=gamma*sum(sum((d.*S).^2));

% ftrace=alpha*trace(S*L*S');

fregs = 0;
for ii = size(S, 1),
    fregs = fregs + beta*S(:, ii)'*Sigma*S(:, ii);
end

fobj = fresidue + fsparsity + fregs;