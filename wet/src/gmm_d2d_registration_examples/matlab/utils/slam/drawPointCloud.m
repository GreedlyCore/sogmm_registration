function [] = drawPointCloud(pose, pcld)
  idxs = find(vecnorm(pcld') > 15);
  pcld(idxs,:) = [];

  idxs = find(pcld(:,3) > -1);
  pcld(idxs,:) = [];

  R = eye(3);
  t = zeros(3,1);

  if (size(pose,2) == 6)
      R = ZYXToR(pose(4:6));
      t = pose(1:3)';
  else
      R = pose(1:3,1:3);
      t = pose(1:3,4);
  end

  pcld = transpose(R*pcld' + repmat(t, 1, length(pcld)));
  pcld = pcdownsample(pointCloud(pcld), 'gridAverage', 0.05);
  pcshow(pcld);
end