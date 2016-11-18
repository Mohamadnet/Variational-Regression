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

%% settings

D_ML = 1;
Ds = 1:7;
x_range = [-5 5];
r=xlsread('ANN.xlsx');
x = r(:,1:size(r,2)-1);
y = r(:,size(r,2));

PointNum = 12;
k=1;
x_test = [];
x_axis = [1];
for i=1:size(x,1)-1
    for j=1:size(x,2)
        x_test(k : k+PointNum , j) = linspace(x(i,j),x(i+1,j), PointNum+1)';
    end
    k = k+PointNum;
    x_axis = [x_axis k];
end
x_test_axis = 1:size (x_test,1);

%separate train and test records
train_record_num = 1:floor(size(x,1)*0.7);
test_record_num = floor(size(x,1)*0.7)+1:size(x,1);
train_record = x(train_record_num,:);
test_record = x(test_record_num,:);

%gen_X = @(x, d) bsxfun(@power, x, 0:(d-1));
%X = gen_X(x, D);
% X = ones(size(x(:,1)));
% for i=1:D
%    X = [X , x.^i]; 
% end
% y = X * w+ randn(N, 1);
% X_test = ones(size(x_test(:,1)));
% for i=1:D
%    X_test = [X_test , x_test.^i]; 
% end
% y_test = X_test * w;


%% perform model selection
Ls = NaN(1, length(Ds));
% % M = ones(size(x(:,1)));
% % for i = 1:length(Ds)
% %     M = [M , x.^i];
% %     [~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit_ard(M, y );
% % end
% % [~, i] = max(Ls);
% % D_best = Ds(i);
%% train section
Ls_train = NaN(1, length(Ds));
M = ones(size(train_record(:,1)));
for i = 1:length(Ds)
    M = [M , train_record.^i];
    [~, ~, ~, ~, ~, ~, ~, Ls_train(i)] = vb_linear_fit_ard(M, y(train_record_num));
end
[~, i] = max(Ls_train);
D_best_train = Ds(i);


%% predictions for selected model
% % % variational bayes
% % M = ones(size(x(:,1)));
% % X_test = ones(size(x_test(:,1)));
% % for i = 1:D_best
% %     M = [M , x.^i];
% %     X_test = [X_test , x_test.^i]; 
% % end
% % [w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit_ard(M, y );
% %     [y_VB, lam_VB, nu_VB] = ...
% %     vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB);
% variational bayes
M = ones(size(train_record(:,1)));
X_test = ones(size(test_record(:,1)));
X_test_2 = ones(size(x(:,1)));
for i = 1:D_best_train
    M = [M , train_record.^i];
    X_test = [X_test , test_record.^i]; 
    X_test_2 = [X_test_2 , x.^i]; 
end
[w_VB_test, V_VB_test, ~, ~, an_VB_test, bn_VB_test] = vb_linear_fit_ard(M, y(train_record_num) );
    [y_VB_test, lam_VB_test, nu_VB_test] = ...
    vb_linear_pred(X_test, w_VB_test, V_VB_test, an_VB_test, bn_VB_test);

    [y_VB, lam_VB, nu_VB] = ...
    vb_linear_pred(X_test_2, w_VB_test, V_VB_test, an_VB_test, bn_VB_test);
%[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(gen_X(x, D_best), y);

%[y_VB, lam_VB, nu_VB] = ...
 %   vb_linear_pred(gen_X(x_test, D_best), w_VB, V_VB, an_VB, bn_VB);
% % y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
% maximum likelihood
% % M = ones(size(x(:,1)));
% % X_test = ones(size(x_test(:,1)));
% % for i = 1:D_ML
% %     M = [M , x.^i];
% %     X_test = [X_test , x_test.^i]; 
% % end
% % w_ML = regress(y, M);
% % y_ML = X_test * w_ML;
M = ones(size(train_record(:,1)));
X_test = ones(size(test_record(:,1)));
X_test_2 = ones(size(x(:,1)));
for i = 1:D_ML
    M = [M , train_record.^i];
    X_test = [X_test , test_record.^i]; 
    X_test_2 = [X_test_2 , x.^i]; 
end
w_ML = regress(y(train_record_num), M);
y_ML_test = X_test * w_ML;
%w_ML = regress(y, gen_X(x, D_ML));
y_ML = X_test_2 * w_ML;
% prediction error
fprintf('Test set MSE, ML = %f, VB = %f\n', ...
        mean((y(test_record_num) - y_ML_test).^2), mean((y(test_record_num) - y_VB_test).^2));


%% plot model selection result
f1 = figure;  hold on;
plot(Ds, Ls_train, 'k-', 'LineWidth', 1);
plot([1 1] * (D_best_train ), ylim, 'k--', 'LineWidth', 0.5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('polynomial order');
ylabel('variational bound');


%% plot prediction
f2 = figure;  hold on;
% shaded CI area
patch([x_axis'; flipud(x_axis')], ...
     [(y_VB + y_VB_sd); flipud(y_VB - y_VB_sd)], ...
     [1 1 1] * 0.9, 'EdgeColor', 'none');
% true and esimtated outputs
%plot(x_test, y_test, 'k-', 'LineWidth', 1);
plot(x_axis, y_VB, '--', 'Color', [0.8 0 0], 'LineWidth', 1);
plot(x_axis, y_ML, '-.', 'Color', [0 0 0.8], 'LineWidth', 1);
x = linspace(x_range(1)+1, x_range(2)-1, length(y))';
plot(x_axis, y, 'k+', 'MarkerSize', 5);
%set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
 %   'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('x');
ylabel('y, y_{ML}, y_{VB}');


