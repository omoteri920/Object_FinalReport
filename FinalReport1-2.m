% BoFベクトル + 非線形SVM

addpath(".")

each_n = 100;
n = each_n * 2;
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

% ここまで画像の読み込み

Trainings = {images_A{:}, images_B{:}};

% コードブックの作成
codebook = mk_codebook(Trainings);
%load('codebook.mat','codebook');
size(codebook)

% BoFベクトルの作成
bof = mk_code(Trainings, codebook);
%load('bof.mat', 'bof');

size(bof)

pos_bof = bof(1:100, :); 
neg_bof = bof(101:200, :);

size(pos_bof)

accuracy = [];

% 5-fold cross validationで分類率を計算
for i = 1:cv_n
    train_pos = pos_bof(find(mod([1:each_n],cv_n) ~= (i-1)),:);
    eval_pos  = pos_bof(find(mod([1:each_n],cv_n) == (i-1)),:);
    train_neg = neg_bof(find(mod([1:each_n],cv_n) ~= (i-1)),:);
    eval_neg  = neg_bof(find(mod([1:each_n],cv_n) == (i-1)),:);

    train = [train_pos; train_neg];
    eval  = [eval_pos; eval_neg];

    train_label=[ones(size(train_pos, 1),1); ones(size(train_neg, 1),1)*(-1)];
    eval_label =[ones(size(eval_pos, 1),1); ones(size(eval_neg, 1),1)*(-1)];

    model = fitcsvm(train, train_label, 'KernelFunction','rbf', 'KernelScale','auto');
    [plabel, score] = predict(model, eval);
    
    ac = numel(find(eval_label==plabel)) / numel(eval_label);
    accuracy = [accuracy ac];
end

fprintf('accuracy: %f\n', mean(accuracy));