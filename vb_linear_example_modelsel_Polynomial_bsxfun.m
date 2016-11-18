%% model selection with vb_linear_*
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed to re-produce JSS figures
rng(1);
%phi=@(a)(bsxfun(@power,a,[0:7]));   Polynomial Regression
%phi=@(a)(2*[cos(bsxfun(@times,a/8,[0:8])),sin(bsxfun(@times,a/8,[1:8]))]);FourierRegression
%phi=@(a)(-1+2*bsxfun(@lt,a,linspace(-8,8,16))); Step Regression
%phi=@(a)(bsxfun(@gt,a,linspace(-8,8,16))); Step Regression II
%phi=@(a)(bsxfun(@minus,abs(bsxfun(@minus,a,linspace(-8,8,16))),linspace(-8,8,16)));VRegression
%phi=@(a)(bsxfun(@times,legendre(13,a/8)',0.15.^[0:13])); LegenreRegression
%phi=@(a)(exp(-abs(bsxfun(@minus,a,[-8:1:8])))); EiffelTowerRegression
%phi=@(a)(exp(-0.5*bsxfun(@minus,a,[-8,1:8].^2))); BellCurveRegression
phi=@(a,F)(bsxfun(@power,a,[0:F]));   %Polynomial Regression
%% settings
D = 3;
N = 20;
D_ML = 3;
Ds = 1:9;
x_range = [-5 5];
X = [];
X_test = [];
% generate data
w = randn(2*(D+1),1);
x = x_range(1) + (x_range(2) - x_range(1)) * rand(N, 1);
x = [x x_range(1) + (x_range(2) - x_range(1)) * rand(N, 1)];
x_test = linspace(x_range(1), x_range(2), 100)';
x_test = [x_test linspace(x_range(1), x_range(2), 100)'];
%gen_X = @(x, d) bsxfun(@power, x, 0:(d-1));
%X = gen_X(x, D);
X = phi(x(:,1),D);
X = [X phi(x(:,2),D)];
y = X * w+ randn(N, 1);
X_test = phi(x_test(:,1),D); 
X_test = [X_test phi(x_test(:,2),D)];
y_test = X_test * w;


%% perform model selection
Ls = NaN(1, length(Ds));
for i = 1:length(Ds)
    M = [];
    M = phi(x(:,1),i);
    M = [M phi(x(:,2),i)];
    [~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit(M, y );
    
end
[~, i] = max(Ls);
D_best = Ds(i);


%% predictions for selected model
% variational bayes
M = [];
M = phi(x(:,1),D_best);
M = [M phi(x(:,2),D_best)];
X_test = [];
X_test = phi(x_test(:,1),D_best); 
X_test = [X_test phi(x_test(:,2),D_best)];
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(M, y );
    [y_VB, lam_VB, nu_VB] = ...
    vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB);
%[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(gen_X(x, D_best), y);

%[y_VB, lam_VB, nu_VB] = ...
 %   vb_linear_pred(gen_X(x_test, D_best), w_VB, V_VB, an_VB, bn_VB);
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
% maximum likelihood
D = D_ML;
M = [];
M = phi(x(:,1),D_ML);
M = [M phi(x(:,2),D_ML)];
X_test = [];
X_test = phi(x_test(:,1),D_ML); 
X_test = [X_test phi(x_test(:,1),D_ML)];
w_ML = regress(y, M);
y_ML = X_test * w_ML;
%w_ML = regress(y, gen_X(x, D_ML));
%y_ML = gen_X(x_test, D_ML) * w_ML;
% prediction error
fprintf('Test set MSE, ML = %f, VB = %f\n', ...
        mean((y_test - y_ML).^2), mean((y_test - y_VB).^2));


%% plot model selection result
figure;
plot3(x(:,1),x(:,2),y,'.r', 'markersize', 20);axis vis3d;
hold on
x1 = x_test(:,1);
y1 = x_test(:,2);
[X1,Y1] = meshgrid(x1, y1);
Z = w_VB(1);
for i=1:D_best
    Z = w_VB(2*i)*X1.^i + w_VB(2*i+1)*Y1.^i + Z;
end
surf(x1, y1, Z);
grid on
xlabel('x1');
ylabel('x2');
zlabel('Variational Bayesian');

hold off
figure;
plot3(x(:,1),x(:,2),y,'.r', 'markersize', 20);axis vis3d;
hold on
[X1,Y1] = meshgrid(x1, y1);
Z = w_ML(1);
for i=1:D_ML
    Z = w_ML(2*i)*X1.^i + w_ML(2*i+1)*Y1.^i + Z;
end
surf(x1, y1, Z);
grid on
xlabel('x1');
ylabel('x2');
zlabel('Maximum Likelihood');
