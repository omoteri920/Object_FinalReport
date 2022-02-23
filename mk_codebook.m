function Ans=mk_codebook(tlist)
  % tlistは学習する画像データ

  % コードブックサイズ
  k=1000;
  
  features=[];
  % 特徴抽出
  for i=1:length(tlist)
    I=rgb2gray(tlist{i});
    % fprintf('reading [%d] %s\n',i,tlist{i});
    %pnt=detectSURFFeatures(I);
    pnt=createRandomPoints(I,1000);
    [fea,vpnt] = extractFeatures(I,pnt);
    features=[features; fea];
  end
 
  size(features);
  [index, codebook]=kmeans(features,k);
  % size(codebook);
  save('codebook.mat','codebook');
  Ans = codebook;
end