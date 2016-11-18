%% model selection with vb_linear_*
%
% Copyright (c) 2014, Jan Drugowitsch
% All rights reserved.
% See the file LICENSE for licensing information.

% Modified by Mohammad Mobasher-Kashani (Mohamadnet)

%% set RNG seed to re-produce JSS figures
% rng(1);
%phi=@(a)(bsxfun(@power,a,[0:7]));   Polynomial Regression
%phi=@(a)(2*[cos(bsxfun(@times,a/8,[0:8])),sin(bsxfun(@times,a/8,[1:8]))]);FourierRegressionOKOKOK
%phi=@(a)(-1+2*bsxfun(@lt,a,linspace(-8,8,16))); Step Regression  ok
%phi=@(a)(bsxfun(@gt,a,linspace(-8,8,16))); Step Regression II
%phi=@(a)(bsxfun(@minus,abs(bsxfun(@minus,a,linspace(-8,8,16))),linspace(-8,8,16)));VRegressionOKOKOK..n=16
%phi=@(a)(bsxfun(@times,legendre(13,a/8)',0.15.^[0:13]));%LegenreRegressionOKOKOK
%phi=@(a)(exp(-abs(bsxfun(@minus,a,[-8:1:8])))); EiffelTowerRegression ok
%phi=@(a)(exp(-0.5*bsxfun(@minus,a,[-8,1:8].^2))); BellCurveRegression
phi=@(a,n)(exp(-abs(bsxfun(@minus,a,[-4:n:4]))));


%%Preprocessing
%z-score standardization
% % r=xlsread('TE_NEW_Spline.xlsx');
% % for i=2:size(r,2)-1
% %     x(:,i-1) = (r(1:size(r,1),i)*min((r(1:size(r,1),i)))/std(r(1:size(r,1),i)));
% % end
% % y = (r(1:size(r,1),size(r,2))*min((r(1:size(r,1),size(r,2))))/std(r(1:size(r,1),size(r,2))));
% % % r=xlsread('TE_NEW_Raz.xlsx');
% % % x = r(:,2:size(r,2)-1);
% % % y = r(:,size(r,2));
%% settings
N = [1/6 1/4 1/2 1 2 3 4];   % model selection experimental design
N_MLE = 3;
PointNum = 1;
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
    for j=2:8
        M = [M phi(train_record(:,j),N(i))];
    end
    %Prior Without ARD 0.01,0.0001,0.01,0.0001
    %Prior With ARD 0.02,0.0001,0.02,0.0001
    [~, ~, ~, ~, ~, ~, ~, Ls(i)] = vb_linear_fit_ard(M, y(train_record_num),0.02,0.0002,0.02,0.0002);
    
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
for i=2:8
    M = [M phi(train_record(:,i),N_best)];
    X_test = [X_test phi(test_record(:,i),N_best)];
    X_test_2 = [X_test_2 phi(x_test(:,i),N_best)];
end

    
[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit_ard(M, y(train_record_num),0.02,0.0002,0.02,0.0002);
    [y_VB_test, lam_VB, nu_VB] = ...
    vb_linear_pred(X_test, w_VB, V_VB, an_VB, bn_VB);
%[w_VB, V_VB, ~, ~, an_VB, bn_VB] = vb_linear_fit(gen_X(x, D_best), y);
[y_VB, lam_VB, nu_VB] = ...
    vb_linear_pred(X_test_2, w_VB, V_VB, an_VB, bn_VB);
%[y_VB, lam_VB, nu_VB] = ...
 %   vb_linear_pred(gen_X(x_test, D_best), w_VB, V_VB, an_VB, bn_VB);
y_VB_sd = sqrt(nu_VB ./ (lam_VB .* (nu_VB - 2)));
% maximum likelihood
% % % Ls_MLE = [];
% % % for i=1:length(N)
% % %     M = [];
% % %     M = phi(train_record(:,1),N(i));
% % %     X_test = [];
% % %     X_test = phi(test_record(:,1),N(i)); 
% % % 
% % %     for j=2:8
% % %         M = [M phi(train_record(:,j),N(i))];
% % %         X_test = [X_test phi(test_record(:,j),N(i))];
% % %         X_test_2 = [X_test_2 phi(x_test(:,j),N(i))];
% % %     end
% % %     w_ML = regress(y(train_record_num), M);
% % %     y_ML_test = X_test * w_ML;
% % %     Ls_MLE(i) = mean((y(test_record_num) - y_ML_test).^2);
% % % end
% % % [~, i] = min(Ls_MLE);
% % % N_MLEbest = N(i);
% % % M = [];
% % % M = phi(train_record(:,1),N_MLEbest);
% % % X_test = [];
% % % X_test = phi(test_record(:,1),N_MLEbest); 
% % % X_test_2 = [];
% % % X_test_2 = phi(x_test(:,1),N_MLEbest); 
% % % for i=2:8
% % %     M = [M phi(train_record(:,i),N_MLEbest)];
% % %     X_test = [X_test phi(test_record(:,i),N_MLEbest)];
% % %     X_test_2 = [X_test_2 phi(x_test(:,i),N_MLEbest)];
% % % end
% % % w_ML = regress(y(train_record_num), M);
% % % y_ML_test = X_test * w_ML;
% % % y_ML = X_test_2 * w_ML;
% % % % prediction error
% % % fprintf('Test set MSE, ML = %f, VB = %f\n', ...
% % %         mean((y(test_record_num) - y_ML_test).^2), mean((y(test_record_num) - y_VB_test).^2));
fprintf('Test set MSE, VB = %f\n', mean((y(test_record_num) - y_VB_test).^2));

%% plot model selection result
f1 = figure;  hold on;
plot(N, Ls, 'k-', 'LineWidth', 1);
plot([1 1] * (N_best), ylim, 'k--', 'LineWidth', 0.5);
set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
    'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('Model order');
ylabel('variational bound');
X_test_3 = [];
X_test_3 = phi(x(:,1),N_best); 
for i=2:8
    X_test_3 = [X_test_3 phi(x(:,i),N_best)];
end
[actual_y_VB, actual_lam_VB, actual_nu_VB] = vb_linear_pred(X_test_3, w_VB, V_VB, an_VB, bn_VB);
actual_y_VB_sd = sqrt(actual_nu_VB ./ (actual_lam_VB .* (actual_nu_VB - 2)));
Rate = 0.5;
for i=1:8
    temp_x = X_test_3;
    temp_x(:,i) = Rate * temp_x(:,i);
    
    X_test_3 = [];
    temp = x(:,1);
    if i==1
        temp = temp + Rate*temp;
    end
    X_test_3 = phi(temp,N_best); 
    for j=2:8
        temp = x(:,j);
        if i==j
            temp = temp + Rate*temp;
        end
        X_test_3 = [X_test_3 phi(temp,N_best)]; 
    end
    
    [temp_y_VB, temp_lam_VB, temp_nu_VB] = vb_linear_pred(X_test_3, w_VB, V_VB, an_VB, bn_VB);
    temp_y_VB_sd = sqrt(temp_nu_VB ./ (temp_lam_VB .* (temp_nu_VB - 2)));
% % %   fprintf('Target Change Rate, VB = %f , sd = %f\n', (mean((y - temp_y_VB).^2)-mean((y - actual_y_VB).^2))/mean((y - actual_y_VB).^2),(mean(temp_y_VB_sd)-mean(actual_y_VB_sd))/mean(actual_y_VB_sd));
    fprintf('Target Change Rate, VB = %f , sd = %f\n', 100*mean((temp_y_VB - actual_y_VB)./actual_y_VB),100*mean((temp_y_VB_sd - actual_y_VB_sd)./actual_y_VB_sd));
end
%% plot prediction
f2 = figure;  hold on;
% shaded CI area
patch([x_test_axis'; flipud(x_test_axis')], ...
     [(y_VB + y_VB_sd); flipud(y_VB - y_VB_sd)], ...
     [1 1 1] * 0.9, 'EdgeColor', 'none');
% true and esimtated outputs
%plot(x_test, y_test, 'k-', 'LineWidth', 1);
plot(x_test_axis, y_VB, '--', 'Color', [0.8 0 0], 'LineWidth', 1);
% % % plot(x_test_axis, y_ML, '-.', 'Color', [0 0 0.8], 'LineWidth', 1);
%%x = linspace(x_range(1)+1, x_range(2)-1, length(y))';
plot(x_axis, y, 'k+', 'MarkerSize', 5);
%set(gca, 'Box','off', 'PlotBoxAspectRatio', [4/3 1 1], ...
 %   'TickDir', 'out', 'TickLength', [1 1]*0.02/(4/3));
xlabel('x');
ylabel('y_{VB}');
