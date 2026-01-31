%% L1QP_FeatureSign solves nonnegative quadradic programming 
%% using Feature Sign. 
%%
%%    min  0.5*x'*A*x+b'*x+\lambda*|x|
%%
%% [net,control]=NNQP_FeatureSign(net,A,b,control)
%%  
%% 
%%

% function [x]=L1QP_FeatureSign_yang(lambda,A,b)
% 
% EPS = 1e-9;
% x=zeros(size(A, 1), 1);           %coeff
% 
% grad=A*sparse(x)+b;
% [ma mi]=max(abs(grad).*(x==0));
% 
% while true,
%     
%     
%   if grad(mi)>lambda+EPS,
%     x(mi)=(lambda-grad(mi))/A(mi,mi);
%   elseif grad(mi)<-lambda-EPS,
%     x(mi)=(-lambda-grad(mi))/A(mi,mi);            
%   else
%     if all(x==0)
%       break;
%     end
%   end    
%   
%   while true,
%     a=x~=0;   %active set
%     Aa=A(a,a);
%     ba=b(a);
%     xa=x(a);
% 
%     %new b based on unchanged sign
%     vect = -lambda*sign(xa)-ba;
%     x_new= Aa\vect;
%     idx = find(x_new);
%     o_new=(vect(idx)/2 + ba(idx))'*x_new(idx) + lambda*sum(abs(x_new(idx)));
%     
%     %cost based on changing sign
%     s=find(xa.*x_new<=0);
%     if isempty(s)
%       x(a)=x_new;
%       loss=o_new;
%       break;
%     end
%     x_min=x_new;
%     o_min=o_new;
%     d=x_new-xa;
%     t=d./xa;
%     for zd=s',
%       x_s=xa-d/t(zd);
%       x_s(zd)=0;  %make sure it's zero
% %       o_s=L1QP_loss(net,Aa,ba,x_s);
%       idx = find(x_s);
%       o_s = (Aa(idx, idx)*x_s(idx)/2 + ba(idx))'*x_s(idx)+lambda*sum(abs(x_s(idx)));
%       if o_s<o_min,
%         x_min=x_s;
%         o_min=o_s;
%       end
%     end
%     
%     x(a)=x_min;
%     loss=o_min;
%   end 
%     
%   grad=A*sparse(x)+b;
%   
%   [ma mi]=max(abs(grad).*(x==0));
%   if ma <= lambda+EPS,
%     break;
%   end
% end

function [x]=L1QP_FeatureSign_yang(X,U)

% 
[dFea, nSmp] = size(X);
nBases = size(U, 2);
Sigma = 100;
z_label = nBases;
y_label = nSmp; 
K = 3;

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



