classdef PoseMsgs
  properties
    positions; %3xn matrix
    orientations; %4xn matrix
    linear_velocity; %3xn matrix
    angular_velocity; %3xn matrix
    times;
  end
  methods
    function this = PoseMsgs(varargin)

	msgs = varargin{1};
	bagfile = varargin{2};

      try

	% First, try to load with matlab_rosbag
	bag = ros.Bag.load(bagfile);
	if (findstr(msgs, '/tf'))
          tf_msgs = bag.readAll({'/tf'});
          child_str = strrep(msgs, '/tf', '');
          accessor = @(x) x.transforms.child_frame_id;
          child_frames = cellfun(accessor, tf_msgs, 'UniformOutput', false);
          pose_idxs = find(strcmp(child_frames, child_str) == 1);
          pose_tfs = tf_msgs(pose_idxs);
          accessor = @(x) x.transforms.transform.translation;
          this.positions = ros.msgs2mat(pose_tfs, accessor);
          accessor = @(x) x.transforms.transform.rotation;
          this.orientations = ros.msgs2mat(pose_tfs, accessor);
          accessor = @(x) x.transforms.header.stamp.time;
          this.times = ros.msgs2mat(pose_tfs, accessor);
	else

	  % geometry_msgs/PoseStamped
          pose_msgs = bag.readAll(msgs);
          try
            accessor = @(x) x.pose.position;
            this.positions = ros.msgs2mat(pose_msgs, accessor);
            accessor = @(x) x.pose.orientation;
            this.orientations = ros.msgs2mat(pose_msgs, accessor);
            accessor = @(x) x.header.stamp.time;
            this.times = ros.msgs2mat(pose_msgs,accessor);
            return
          catch
            this.positions = [];
            this.orientations = [];
            this.times = [];
          end
          try
	    %  nav_msgs/Odometry
            accessor = @(x) x.pose.pose.position;
            this.positions = ros.msgs2mat(pose_msgs, accessor);
            accessor = @(x) x.pose.pose.orientation;
            this.orientations = ros.msgs2mat(pose_msgs, accessor);
            accessor = @(x) x.header.stamp.time;
            this.times = ros.msgs2mat(pose_msgs, accessor);
            accessor = @(x) x.twist.twist.linear;
            this.linear_velocity = ros.msgs2mat(pose_msgs, accessor);
            accessor = @(x) x.twist.twist.angular;
            this.angular_velocity = ros.msgs2mat(pose_msgs, accessor);
            return
          catch
            this.positions  = [];
            this.orientations = [];
            this.times = [];
          end
	end
      catch

	% If that doesn't work, use MATLAB's built-in msgs files
	bagselect = rosbag(bagfile);
	start_time = 0;
	end_time = bagselect.EndTime - bagselect.StartTime;

	if (nargin > 2)
	  start_time = varargin{3};
	  end_time = varargin{4};
	end

	bagselect2 = select(bagselect, 'Topic', msgs, 'Time', [bagselect.StartTime + start_time, bagselect.StartTime + end_time]);
	poses = readMessages(bagselect2);
	this.times = cellfun(@(x) double(x.Header.Stamp.Sec)+double(x.Header.Stamp.Nsec)/1e9, poses);
	this.positions = cell2mat(cellfun(@(x) double([x.Pose.Pose.Position.X; x.Pose.Pose.Position.Y; x.Pose.Pose.Position.Z]), poses, 'uni', false)');
	this.orientations = cell2mat(cellfun(@(x) double([x.Pose.Pose.Orientation.X; x.Pose.Pose.Orientation.Y; ...
							  x.Pose.Pose.Orientation.Z; x.Pose.Pose.Orientation.W]), poses, 'uni', false)');
	this.linear_velocity = cell2mat(cellfun(@(x) double([x.Twist.Twist.Linear.X; x.Twist.Twist.Linear.Y; x.Twist.Twist.Linear.Z]), poses, 'uni', false)');
	this.angular_velocity = cell2mat(cellfun(@(x) double([x.Twist.Twist.Angular.X; x.Twist.Twist.Angular.Y; x.Twist.Twist.Angular.Z]), poses, 'uni', false)');

      end
    end
    function [diff, idx] = closestTime(this, time)
      [diff, idx] = min(abs(time-this.times));
    end
  end
end
