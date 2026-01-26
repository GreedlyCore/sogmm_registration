function h = plotGTSAMTrajectory(result, hin)

  if (~isempty(hin))
      delete(hin);
  end

  hold on;

  @(x) result.atPose(x).matrix();
  keys = 1:double(result.keys().size());
  keys_cell = num2cell(keys);
  poses_cell = cellfun(@(x) result.atPose3(x).matrix, keys_cell, 'uni', false);
  translations = cellfun(@(x) x(1:3,4), poses_cell, 'uni', false);
  ts = cell2mat(translations); % returns 3xN vector

  h = plot3(ts(1,:), ts(2,:), ts(3,:), 'Color', [0.75 0 0.75], 'LineWidth', 3);
end