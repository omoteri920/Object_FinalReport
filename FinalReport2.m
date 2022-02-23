addpath('.');

pos_n = 25;
neg_n = 500;
tes_n = 300;

% ここから画像読み込み

list_pos = textread('urllist_sushi50.txt', '%s');
list_test = textread('urllist_sushi_interesting300.txt', '%s');

images_pos = {}; images_neg = {};
images_test = {};

for i = 1:pos_n
    images_pos{i} = webread(list_pos{i});
end

dir_name = 'bgimg/';
DIR = dir(dir_name);
for i = 3:neg_n+3
    filename = strcat(dir_name, DIR(i).name);
    images_neg{i} = imread(filename);
end

for i = 1:tes_n
    images_test{i} = webread(list_test{i});
end

% size(images_pos)
% size(images_neg)
% size(images_test)

% ここまで画像読み込み

Trainings = {images_pos{:}, images_neg{3:neg_n+2}};

size(Trainings)

% save('Trainings.mat', 'Trainings');
% load('Trainings.mat');

% alexnetを使用
net = alexnet;

D_trainings = [];
D_testings  = [];

% 学習用データのDCNN特徴抽出
for i = 1:size(Trainings, 2)
    img = Trainings{i};
    reimg = imresize(img, net.Layers(1).InputSize(1:2));
    dcnnf = activations(net, reimg, 'fc7');
    dcnnf = squeeze(dcnnf);
    dcnnf = dcnnf/norm(dcnnf);
    D_trainings = [D_trainings; dcnnf.'];
end

% テスト用データのDCNN特徴抽出
for i = 1:size(images_test, 2)
    img = images_test{i};
    reimg = imresize(img, net.Layers(1).InputSize(1:2));
    dcnnf = activations(net, reimg, 'fc7');
    dcnnf = squeeze(dcnnf);
    dcnnf = dcnnf/norm(dcnnf);
    D_testings = [D_testings; dcnnf.'];
end

%学習と予測
train_label=[ones(pos_n,1); ones(neg_n,1)*(-1)];
model = fitcsvm(D_trainings, train_label, "KernelFunction","linear");
[plabel, score] = predict(model, D_testings);

size(D_trainings);
size(train_label);

% save('Finel_score.mat', 'score');

[sorted_score,sorted_idx] = sort(score(:,2),'descend');

% 適当な枚数を表示
for i=16:30
  %fprintf('%s %f\n',list_test{sorted_idx(i)},sorted_score(i));
  subplot(3, 5, i-15);
  imshow(imread(list_test{sorted_idx(i)}));
  %fprintf('%f\n', sorted_score(i));
end
