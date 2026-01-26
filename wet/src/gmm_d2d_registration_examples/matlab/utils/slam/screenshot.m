function [] = screenshot(idx, DIR)
  set(gcf, 'InvertHardCopy', 'off');
  saveas(gcf, [DIR, sprintf('%06d', idx), '.png'], 'png');
end
