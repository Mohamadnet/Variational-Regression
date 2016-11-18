%% model selection with vb_linear_*
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

%%          Radial Basis Function (RBF)
 kerf=@(z)exp(-z.*z/5)/sqrt(2*pi);
% x = 1:100;
% z=kerf((1-x)/10);
% plot(x,z);
% hold on
% for i=1:50
%     z=kerf((i-x)/10);
%     plot(x,z);
% end

%% set RNG seed to re-produce JSS figures
rng(1);


%% settings
D = 7;
N = 50;
D_ML = 6;
Ds = 1:9;
x_range = [-5 5];
RBF_size = 5;
z = linspace(x_range(1), x_range(2), RBF_size)';   %RBF range
% generate data
w = randn(2*D+2,1);
%%
% x = x_range(1) + (x_range(2) - x_range(1)) * rand(N, 1);
% x_RBF = [];
% for i=1:length(x)
%    x_RBF(i,:) = kerf((x(i)-z)/2);
% end
% x_temp = x_range(1) + (x_range(2) - x_range(1)) * rand(N, 1);
% x = [x x_temp];
%     x_RBF_temp = [];
%     for i=1:length(x)
%        x_RBF_temp(i,:) = kerf((x(i)-z)/2);
%     end
% x_RBF = [x_RBF x_RBF_temp];
r=xlsread('ANN.xlsx');
x = r(:,1:size(r,2)-1);
y = r(:,size(r,2));
x_RBF = [];
for j=1:size(x,2)
    x_RBF_temp = [];
    for i=1:size(x,1)
       x_RBF_temp(i,:) = kerf((x(i,j)-z)/2);
    end
    x_RBF = [x_RBF x_RBF_temp];
end
%%
x_test = linspace(x_range(1), x_range(2), 100)';
x_test_RBF = [];
for i=1:size(x_test,1)
   x_test_RBF(i,:) = kerf((x_test(i)-z)/2);
end
x_test = [x_test x_test x_test x_test x_test x_test x_test x_test];
x_test_RBF = [x_test_RBF x_test_RBF x_test_RBF x_test_RBF x_test_RBF x_test_RBF x_test_RBF x_test_RBF];

%gen_X = @(x, d) bsxfun(@power, x, 0:(d-1));
%X = gen_X(x, D);


% w_RBF=[];
% w_RBF_temp = [];
% for i=1:length(w)
%     for j=1:RBF_size
%         w_RBF_temp(j) = w(i);
%     end
%     w_RBF=[w_RBF;w_RBF_temp'];
% end

X = ones(size(x_RBF));
X_test = ones(size(x_test_RBF));
for i=1:D
    X = [X , x_RBF.^i]; 
    X_test = [X_test , x_test_RBF.^i]; 
end
w_RBF = rand (size(X,2),1);
y = X * w_RBF+ rand(N, 1);
y_test = X_test * w_RBF;


%% perform model selection
Ls = NaN(1, length(Ds));
M = ones(size(x_RBF));
%X = ones(size(x_expo_temp));
mode = 2; %multivariate mode
for i = 1:length(Ds)
    M = [M , x_RBF.^i];
    %X = [X , x_expo_temp.^i];
    %[~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit(gen_X(x, Ds(i)), y);
    %[~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit_MV(N, X, y , RBF_size );
    [~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit(M, y );
    
end
[~, i] = max(Ls);
D_best = Ds(i);


%% predictions for selected model
% variational bayes

M = ones(size(x_RBF));
X_test = ones(size(x_test_RBF));
for i = 1:D_best
    M = [M , x_RBF.^i];
    X_test = [X_test , x_test_RBF.^i]; 
end
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(M , y);

% w_RBF=[];
% w_RBF_temp = [];
% for i=1:length(w_VB)
%     for j=1:RBF_size
%         w_RBF_temp(j) = w_VB(i);
%     end
%     w_RBF=[w_RBF;w_RBF_temp'];
% end
    [y_VB, lam_VB, nu_VB] = ...
    vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB);
%[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(gen_X(x, D_best), y);

%[y_VB, lam_VB, nu_VB] = ...
 %   vb_linear_pred(gen_X(x_test, D_best), w_VB, V_VB, an_VB, bn_VB);
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
% maximum likelihood
M = ones(size(x));
X_test = ones(size(x_test));
for i = 1:D_ML
    M = [M , x.^i];
    X_test = [X_test , x_test.^i]; 
end
w_ML = regress(y, M);
y_ML = X_test * w_ML;
%w_ML = regress(y, gen_X(x, D_ML));
%y_ML = gen_X(x_test, D_ML) * w_ML;
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
%patch([x_test; flipud(x_test)], ...
 %     [(y_VB + 1.96 * y_VB_sd); flipud(y_VB - 1.96 * y_VB_sd)], ...
  %    [1 1 1] * 0.9, 'EdgeColor', 'none');
% true and esimtated outputs
plot(x_test(:,1), y_test, 'k-', 'LineWidth', 1);
plot(x_test(:,1), y_VB, '--', 'Color', [0.8 0 0], 'LineWidth', 1);
plot(x_test(:,1), y_ML, '-.', 'Color', [0 0 0.8], 'LineWidth', 1);
plot(x(:,1), y, 'k+', 'MarkerSize', 5);
%set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
 %   'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('x');
ylabel('y, y_{ML}, y_{VB}');
