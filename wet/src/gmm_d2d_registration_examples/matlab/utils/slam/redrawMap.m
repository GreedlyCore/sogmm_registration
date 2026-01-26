function [] = redrawMap(poses, idxs, pclds)
  close all;
  figure; hold on;
  for i = 1:length(idxs)
      drawPointCloud(poses(i,:), pclds.points{idxs(i)});
  end
end