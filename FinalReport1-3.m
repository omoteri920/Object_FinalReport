% AlexNetによるDCNN特徴 + 線形SVM

addpath(".")

each_n = 100;
n = each_n *2;
cv_n = 5;

% ここから画像読み込み

%list_A = textread('urllist_pasta.txt', '%s');
%list_B = textread('urllist_curry.txt', '%s');
list_A = textread('urllist_ramen.txt', '%s');
list_B = textread('urllist_aburasoba.txt', '%s');

images_A = {}; images_B = {};

for i = 1:each_n
    images_A{i} = webread(list_A{i});
    images_B{i} = webread(list_B{i});
end

Trainings = {images_A{:}, images_B{:}};

% Alexnetを使用
net = alexnet;
D = [];

% DCNN特徴の抽出
for i = 1:n
    img = Trainings{i};
    reimg = imresize(img, net.Layers(1).InputSize(1:2));
    dcnnf = activations(net, reimg, 'fc7');
    dcnnf = squeeze(dcnnf);
    dcnnf = dcnnf/norm(dcnnf);
    D = [D; dcnnf.'];
end

size(D);

pos_D = D(1:100, :);
neg_D = D(101:200, :);

accuracy = [];

% 5-fold cross validationで分類率を計算
for i = 1:cv_n
    train_pos = pos_D(find(mod([1:each_n],cv_n) ~= (i-1)),:);
    eval_pos  = pos_D(find(mod([1:each_n],cv_n) == (i-1)),:);
    train_neg = neg_D(find(mod([1:each_n],cv_n) ~= (i-1)),:);
    eval_neg  = neg_D(find(mod([1:each_n],cv_n) == (i-1)),:);

    train = [train_pos; train_neg];
    eval  = [eval_pos; eval_neg];

    train_label=[ones(size(train_pos, 1),1); ones(size(train_neg, 1),1)*(-1)];
    eval_label =[ones(size(eval_pos, 1),1); ones(size(eval_neg, 1),1)*(-1)];

    model = fitcsvm(train, train_label, "KernelFunction","linear");
    [plabel, score] = predict(model, eval);
    
    ac = numel(find(eval_label==plabel)) / numel(eval_label);
    accuracy = [accuracy ac];
end

fprintf('accuracy: %f\n', mean(accuracy));