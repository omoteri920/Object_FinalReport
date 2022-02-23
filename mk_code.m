function code=mk_code(tlist, codebook)
  % tlistは学習する画像データ

  k=size(codebook,1);
  dim=size(codebook,2);
 
  code=[];
 
  % BoFベクトルの作成
  for i=1:length(tlist)
    c=zeros(k,1);
    I=rgb2gray(tlist{i});
    pnt=detectSURFFeatures(I);
    pnt=createRandomPoints(I,2000);
    [fea,vpnt] = extractFeatures(I,pnt);
 
    for j=1:size(fea,2)
      s=zeros(1,k);
      for t=1:dim
        s=s+(codebook(:,t)-fea(j,t)).^2;
      end
      [dist,sidx]=min(s);
      c(sidx,1)=c(sidx,1)+1.0;
    end
    %正規化
    c=c/sum(c); 
    code=[code c];
  end
  fprintf('bof size is: ');
  size(code)
  code = code.';
  save("bof.mat", 'code');
end