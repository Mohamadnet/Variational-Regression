%% model selection with vb_linear_*
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.


%% set RNG seed to re-produce JSS figures
rng(1);
%phi=@(a)(bsxfun(@power,a,[0:7]));   Polynomial Regression
%phi=@(a)(2*[cos(bsxfun(@times,a/8,[0:8])),sin(bsxfun(@times,a/8,[1:8]))]);FourierRegressionOKOKOK
%phi=@(a)(-1+2*bsxfun(@lt,a,linspace(-8,8,16))); Step Regression  ok
%phi=@(a)(bsxfun(@gt,a,linspace(-8,8,16))); Step Regression II
%phi=@(a)(bsxfun(@minus,abs(bsxfun(@minus,a,linspace(-8,8,16))),linspace(-8,8,16)));VRegressionOKOKOK
%phi=@(a)(bsxfun(@times,legendre(13,a/8)',0.15.^[0:13]));%LegenreRegressionOKOKOK
%phi=@(a)(exp(-abs(bsxfun(@minus,a,[-8:1:8])))); EiffelTowerRegression ok
%phi=@(a)(exp(-0.5*bsxfun(@minus,a,[-8,1:8].^2))); BellCurveRegression

%% settings
D = 5;
N = 10;
D_ML = 4;
Ds = 1:10;
x_range = [-5 5];
% generate data
gen_X = @(a)(bsxfun(@times,legendre(13,a/8)',0.15.^[0:13]));
w = randn(size(gen_X(2),2), 1);
x = x_range(1) + (x_range(2) - x_range(1)) * rand(N, 1);
x_test = linspace(x_range(1), x_range(2), 100)';

X = gen_X(x);
y = X * w + randn(N, 1);
y_test = gen_X(x_test) * w;


%% perform model selection
Ls = NaN(1, length(Ds));
for i = 1:length(Ds)
    [~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit(gen_X(x), y);
end
[~, i] = max(Ls);
D_best = Ds(i);


%% predictions for selected model
% variational bayes
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(gen_X(x), y);
[y_VB, lam_VB, nu_VB] = ...
    vb_linear_pred(gen_X(x_test), w_VB, V_VB, an_VB, bn_VB);
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
% maximum likelihood
w_ML = regress(y, gen_X(x));
y_ML = gen_X(x_test) * w_ML;
% prediction error
fprintf('Test set MSE, ML = %f, VB = %f\n', ...
        mean((y_test - y_ML).^2), mean((y_test - y_VB).^2));


%% plot model selection result
f1 = figure;  hold on;
plot(Ds-1, Ls, 'k-', 'LineWidth', 1);
plot([1 1] * (D - 1), ylim, 'k--', 'LineWidth', 0.5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('polynomial order');
ylabel('variational bound');


%% plot prediction
f2 = figure;  hold on;
% shaded CI area
patch([x_test; flipud(x_test)], ...
      [(y_VB + y_VB_sd); flipud(y_VB - y_VB_sd)], ...
      [1 1 1] * 0.9, 'EdgeColor', 'none');
% true and esimtated outputs
plot(x_test, y_test, 'k-', 'LineWidth', 1);
plot(x_test, y_VB, '--', 'Color', [0.8 0 0], 'LineWidth', 1);
plot(x_test, y_ML, '-.', 'Color', [0 0 0.8], 'LineWidth', 1);
plot(x, y, 'k+', 'MarkerSize', 5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('x');
ylabel('y, y_{ML}, y_{VB}');
