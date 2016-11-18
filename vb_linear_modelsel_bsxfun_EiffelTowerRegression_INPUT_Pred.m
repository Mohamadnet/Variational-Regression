%% model selection with vb_linear_*
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

% Modified by Mohammad Mobasher-Kashani (Mohamadnet)

%% set RNG seed to re-produce JSS figures
phi=@(a,n)(exp(-abs(bsxfun(@minus,a,[1363:n:2013]))));
%% settings
N = [1/6 1/4 1/2 1 2 3 4];   % model selection experimental design
N_MLE = 3;
PointNum = 12;
k=1;
x_test = [];
% % x_axis = [1];
for i=1:size(x,1)-1    %input data linearly change from one recod to the next record
    for j=1:size(x,2)
        x_test(k : k+PointNum , j) = linspace(x(i,j),x(i+1,j), PointNum+1)';
    end
    k = k+PointNum;
% %     x_axis = [x_axis k];
end
% % x_test_axis = 1:size (x_test,1);
x_axis = 1963:2013;
x_test_axis = 1963:1/PointNum:2013;
%separate train and test records
train_record_num = 1:floor(size(x,1)*0.7);
test_record_num = floor(size(x,1)*0.7)+1:size(x,1);
train_record = x(train_record_num,:);
test_record = x(test_record_num,:);


%% perform model selection
Ls = [];
for i = 1:length(N)
    M = [];
    M = phi(train_record(:,1),N(i));
    [~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit_ard(M, y(train_record_num),2,0.2,10,1);
    
end
[~, i] = max(Ls);
N_best = N(i);


%% predictions for selected model
% variational bayes
M = [];
M = phi(train_record(:,1),N_best);
X_test = [];
X_test = phi(test_record(:,1),N_best); 
X_test_2 = [];
X_test_2 = phi(x_test(:,1),N_best); 
    
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit_ard(M, y(train_record_num),2,0.2,10,1);
    [y_VB_test, lam_VB, nu_VB] = ...
    vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB);
%[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(gen_X(x, D_best), y);
[y_VB, lam_VB, nu_VB] = ...
    vb_linear_pred(X_test_2, w_VB, V_VB, an_VB, bn_VB);
%[y_VB, lam_VB, nu_VB] = ...
 %   vb_linear_pred(gen_X(x_test, D_best), w_VB, V_VB, an_VB, bn_VB);
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
fprintf('Test set MSE, VB = %f\n', mean((y(test_record_num) - y_VB_test).^2));

%% plot model selection result
f1 = figure;  hold on;
plot(N, Ls, 'k-', 'LineWidth', 1);
plot([1 1] * (N_best), ylim, 'k--', 'LineWidth', 0.5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('Model order');
ylabel('variational bound');


%% plot prediction
f2 = figure;  hold on;
% shaded CI area
patch([x_test_axis'; flipud(x_test_axis')], ...
     [(y_VB + y_VB_sd); flipud(y_VB - y_VB_sd)], ...
     [1 1 1] * 0.9, 'EdgeColor', 'none');
plot(x_test_axis, y_VB, '--', 'Color', [0.8 0 0], 'LineWidth', 1);

plot(x_axis, y, 'k+', 'MarkerSize', 5);
xlabel('x');
ylabel('y_{VB}');
