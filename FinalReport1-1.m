% カラーヒストグラム + 線形SVM

addpath(".")

each_n = 100;
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

% ここまで画像読み込み

pos_hist = []; neg_hist = [];

% カラーヒストグラムの抽出
for i = 1:each_n
    imgA = images_A{i};
    imgB = images_B{i};

    redA = imgA(:, :, 1); redB = imgB(:, :, 1);
    greA = imgA(:, :, 2); greB = imgB(:, :, 2);
    bluA = imgA(:, :, 3); bluB = imgB(:, :, 3);

    A64 = floor(double(redA)/64)*4*4 + floor(double(greA)/64)*4 + floor(double(bluA)/64);
    B64 = floor(double(redB)/64)*4*4 + floor(double(greB)/64)*4 + floor(double(bluB)/64);

    A64_vec = reshape(A64, 1, numel(A64));
    B64_vec = reshape(B64, 1, numel(B64));

    histA = histc(A64_vec, [0:63]);
    histB = histc(B64_vec, [0:63]);
    histA = histA/sum(histA);
    histB = histB/sum(histB);

    pos_hist = [pos_hist; histA];
    neg_hist = [neg_hist; histB];
end

accuracy = [];

% 5-fold cross validationで分類率を計算
for i = 1:cv_n
    train_pos = pos_hist(find(mod([1:each_n],cv_n) ~= (i-1)),:);
    eval_pos  = pos_hist(find(mod([1:each_n],cv_n) == (i-1)),:);
    train_neg = neg_hist(find(mod([1:each_n],cv_n) ~= (i-1)),:);
    eval_neg  = neg_hist(find(mod([1:each_n],cv_n) == (i-1)),:);

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