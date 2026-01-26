function [idxs] = findPotentialLoopClosures(values, curr_pose, range)

  % get keys
  keys = 1:double(values.keys().size());
  keys_cell = num2cell(keys);

  % get poses
  poses_cell = cellfun(@(x) values.atPose3(x).matrix(), keys_cell, 'uni', false);
  line_items = cellfun(@(x) [x(1:3,4)' RToZYX(x(1:3,1:3))']', poses_cell, 'uni', false);
  poses = cell2mat(line_items)'; % returns 3xN vector

  curr_translation = curr_pose(1:3,4); %3x1

  idxs = find(vecnorm(poses(:,1:3)' - curr_translation) < range);
end