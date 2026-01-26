function [transforms, ground_truths, errors] = read_pkl_file(pkl_filename)

  fid = py.open(pkl_filename, 'rb');
  data = py.pickle.load(fid);
  celldata = cell(data);

  transforms = extractCellData(celldata{1});
  ground_truths = extractCellData(celldata{2});
  errors = extractCellData(celldata{3});
end

function ret = extractCellData(data)
  celldata = cell(data.tolist());
  dim = size(celldata);

  ret = [];
  for i=1:dim(2)
    d = celldata{i};
    ret(end+1,:) = cellfun(@double, cell(d));
  end
end
