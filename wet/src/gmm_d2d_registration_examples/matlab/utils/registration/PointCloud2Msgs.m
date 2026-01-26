classdef PointCloud2Msgs
  properties
    points;
    times;
    rgb;
    intensity;
    ring
  end
  methods
    function this = PointCloud2Msgs(msgs, bag)
      bagselect = rosbag(bag);
      bagselect2 = select(bagselect, 'Time', ...
                          [bagselect.StartTime bagselect.EndTime], 'Topic', msgs);
      pointclouds = readMessages(bagselect2);
      this.times = cellfun(@(x) double(x.Header.Stamp.Sec)+double(x.Header.Stamp.Nsec)/1e9, pointclouds);
      this.points = cellfun(@(x) double(x.readXYZ), pointclouds, 'UniformOutput', false);
      try
        this.rgb = cellfun(@(x) double(x.readRGB*255), pointclouds, 'UniformOutput', false);
      catch
      end
      try
        this.intensity = cellfun(@(x) x.readField('intensity'), pointclouds, 'UniformOutput', false);
      catch
      end
      try
        this.ring = cellfun(@(x) x.readField('ring'), pointclouds, 'UniformOutput', false);
      catch
      end
    end
  end
end
