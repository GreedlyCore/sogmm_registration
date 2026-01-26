function [poses, keys] = readPoseGraphEntries(result)

  % get keys
  keys = 1:double(result.keys().size());
  keys_cell = num2cell(keys);

  % get poses
  poses_cell = cellfun(@(x) result.atPose3(x).matrix(), keys_cell, 'uni', false);
  line_items = cellfun(@(x) [x(1:3,4)' RToZYX(x(1:3,1:3))']', poses_cell, 'uni', false);
  poses = cell2mat(line_items)'; % returns 3xN vector
end