#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/PolyTrajArray.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <quadrotor_msgs/StatuArray.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <visualization_msgs/Marker.h>
#include <traj_opt/poly_traj_utils.hpp>
#include <vis_utils/vis_utils.hpp>

ros::Publisher pos_cmd_pub_, statu_pub_, target_traj_pub_;
ros::Subscriber triger_sub_;
bool receive_traj_ = false;
bool flight_start_ = false;
int drone_num_;
std::shared_ptr<vis_utils::VisUtils> visPtr_;

std::vector<Trajectory> swarm_traj, swarm_traj_last; 

quadrotor_msgs::PolyTrajArray trajMsg_, trajMsg_last_;

Trajectory target_traj;
quadrotor_msgs::PolyTraj target_traj_msg;
ros::Time Trigger_Time;

double Safe_R_;

bool ifLargeScale_;
bool triger_received_ = false;
bool is_end = false;

void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {

    Trigger_Time = ros::Time::now();
    target_traj_msg.start_time = Trigger_Time;
    target_traj_msg.order = 7;

    float *coef_x;
    float *coef_y;
    float *coef_z;
    float coef_x_L[] = {0,0,0,0,0,0,100,1000};
    float coef_y_L[] = {0,0,0,0,0,1.5,-10,10};
    float coef_z_L[] = {0,0,0,0,0,0,10.0,0};
    float coef_x_M[] = {0,0,0,0,0,0,1,5};
    float coef_y_M[] = {0,0,0,0,0,-0.001,0.2,3};
    float coef_z_M[] = {0,0,0,0,0,0,0,5};

    if(ifLargeScale_){
        coef_x = coef_x_L;
        coef_y = coef_y_L;
        coef_z = coef_z_L;
    }
    else{
        coef_x = coef_x_M;
        coef_y = coef_y_M;
        coef_z = coef_z_M;        
    }

    // float coef_x[] = {0,0,0,0,0,0,100,1000};
    // float coef_y[] = {0,0,0,0,0,0,0,0};
    // float coef_z[] = {0,0,0,0,0,0,10.0,0};

    // float coef_x[] = {0,0,0,0,0,0,2,10};
    // float coef_y[] = {0,0,0,0,0,0,0,0};
    // // float coef_z[] = {0,0,0,0,0,-0.1,1.5,3};
    // float coef_z[] = {0,0,0,0,0,0,0,5};
    float duration[] = {50};
    for(int i =0 ; i < 8; i++){
      target_traj_msg.coef_x.push_back(coef_x[i]);
      target_traj_msg.coef_y.push_back(coef_y[i]);
      target_traj_msg.coef_z.push_back(coef_z[i]);
    }
    for(int i = 0;  i< sizeof(duration)/sizeof(int); i++) target_traj_msg.duration.push_back(duration[i]);

    target_traj_pub_.publish(target_traj_msg);

    target_traj.start_time = target_traj_msg.start_time;
    target_traj.order = target_traj_msg.order;
    int piece_nums = target_traj_msg.duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<CoefficientMat> cMats(piece_nums);
    auto& traj = target_traj_msg;
    for (int i = 0; i < piece_nums; ++i) {
      int i6 = i * 8;
      cMats[i].row(0) << traj.coef_x[i6 + 0], traj.coef_x[i6 + 1], traj.coef_x[i6 + 2],
          traj.coef_x[i6 + 3], traj.coef_x[i6 + 4], traj.coef_x[i6 + 5], traj.coef_x[i6 + 6], traj.coef_x[i6 + 7];
      cMats[i].row(1) << traj.coef_y[i6 + 0], traj.coef_y[i6 + 1], traj.coef_y[i6 + 2],
          traj.coef_y[i6 + 3], traj.coef_y[i6 + 4], traj.coef_y[i6 + 5], traj.coef_y[i6 + 6], traj.coef_y[i6 + 7];
      cMats[i].row(2) << traj.coef_z[i6 + 0], traj.coef_z[i6 + 1], traj.coef_z[i6 + 2],
          traj.coef_z[i6 + 3], traj.coef_z[i6 + 4], traj.coef_z[i6 + 5], traj.coef_z[i6 + 6], traj.coef_z[i6 + 7];

      dura[i] = traj.duration[i];
    }
    target_traj.SetTraj(dura, cMats);
    
    triger_received_ = true;
}

void polyTrajCallback(const quadrotor_msgs::PolyTrajArrayConstPtr &msgPtr) {
  drone_num_ = msgPtr->swarm_traj.size();
  swarm_traj.clear();
  swarm_traj.resize(drone_num_);
  swarm_traj_last = swarm_traj;
  if (!receive_traj_) {
    trajMsg_last_ = trajMsg_;
    receive_traj_ = true;
  }

  for(int i = 0; i<drone_num_; i++){
    swarm_traj[i].start_time = msgPtr->swarm_traj[i].start_time;
    swarm_traj[i].order = msgPtr->swarm_traj[i].order;
    int piece_nums = msgPtr->swarm_traj[i].duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<CoefficientMat> cMats(piece_nums);
    auto& traj = msgPtr->swarm_traj[i];
    for (int i = 0; i < piece_nums; ++i) {
      int i6 = i * 8;
      cMats[i].row(0) << traj.coef_x[i6 + 0], traj.coef_x[i6 + 1], traj.coef_x[i6 + 2],
          traj.coef_x[i6 + 3], traj.coef_x[i6 + 4], traj.coef_x[i6 + 5], traj.coef_x[i6 + 6], traj.coef_x[i6 + 7];
      cMats[i].row(1) << traj.coef_y[i6 + 0], traj.coef_y[i6 + 1], traj.coef_y[i6 + 2],
          traj.coef_y[i6 + 3], traj.coef_y[i6 + 4], traj.coef_y[i6 + 5], traj.coef_y[i6 + 6], traj.coef_y[i6 + 7];
      cMats[i].row(2) << traj.coef_z[i6 + 0], traj.coef_z[i6 + 1], traj.coef_z[i6 + 2],
          traj.coef_z[i6 + 3], traj.coef_z[i6 + 4], traj.coef_z[i6 + 5], traj.coef_z[i6 + 6], traj.coef_z[i6 + 7];

      dura[i] = traj.duration[i];
    }
    swarm_traj[i].SetTraj(dura, cMats);
  }
}

bool exe_traj(const std::vector<Trajectory> &swarm_traj) {
  quadrotor_msgs::StatuArray msg;
  std::string id_sample;
  for(int i = 0; i < drone_num_; i++){
    double t = (ros::Time::now() - swarm_traj[i].start_time).toSec();
    if (t > 0) {
        if (t > swarm_traj[i].getTotalDuration()) {
        ROS_WARN("[traj_server] trajectory is to the end!");
        is_end = true;
        return false;
        }

        Eigen::Vector3d p, v, a;
        p = swarm_traj[i].getPos(t);
        v = swarm_traj[i].getVel(t);
        a = swarm_traj[i].getAcc(t);

        quadrotor_msgs::PositionCommand pcmd;
        pcmd.position.x = p.x();
        pcmd.position.y = p.y();
        pcmd.position.z = p.z();
        pcmd.velocity.x = v.x();
        pcmd.velocity.y = v.y();
        pcmd.velocity.z = v.z();
        pcmd.acceleration.x = a.x();
        pcmd.acceleration.y = a.y();
        pcmd.acceleration.z = a.z();

        msg.drone_status.push_back(pcmd);

        id_sample = "optimized_traj_" + std::to_string(i);
        visPtr_->visualize_traj(swarm_traj[i], id_sample);

        id_sample = "drone_dir_" + std::to_string(i);
        std::string id_sample1 = "safe_ball_" + std::to_string(i);
        if(ifLargeScale_){
			visPtr_->visualize_arrow(p, p + 50.0 * v.normalized(), id_sample, vis_utils::yellow,Eigen::Vector3d(10,25,0));
            visPtr_->visualize_a_ball(p,Safe_R_,id_sample1,vis_utils::green, 0.4);
        }
        else{
            visPtr_->visualize_arrow(p, p + 1.0 * v.normalized(), id_sample, vis_utils::yellow, Eigen::Vector3d(0.2,0.3,0));
            visPtr_->visualize_a_ball(p,Safe_R_,id_sample1,vis_utils::green, 0.4);
        }
        continue;
    }
    return false;
  }
  statu_pub_.publish(msg);

  double t = (ros::Time::now() - target_traj.start_time).toSec();
  Eigen::Vector3d target_p = target_traj.getPos(t);
  Eigen::Vector3d target_v = target_traj.getVel(t);

  if(ifLargeScale_)
	visPtr_->visualize_arrow(target_p, target_p + 50.0 * target_v.normalized(), 
							"target_dir", vis_utils::red, Eigen::Vector3d(10,25,0));
  else
	visPtr_->visualize_arrow(target_p, target_p + 1.0 * target_v.normalized(), "target_dir", vis_utils::red, Eigen::Vector3d(0.2,0.3,0));

  visPtr_->visualize_traj(target_traj, "target_traj_zyh");

  return true;
}

void cmdCallback(const ros::TimerEvent &e) {
  if (!receive_traj_) {
    return;
  }

  if(is_end) return;

  if (exe_traj(swarm_traj)) {
    swarm_traj_last = swarm_traj;
    return;

  } else if (exe_traj(swarm_traj_last)) {
    return;
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "sim_node");
  ros::NodeHandle nh("~");

  nh.getParam("ifLargeScale", ifLargeScale_);
  nh.getParam("Safe_R", Safe_R_);

  ros::Subscriber swarm_poly_traj_sub = nh.subscribe("trajectory", 10, polyTrajCallback);
  statu_pub_ = nh.advertise<quadrotor_msgs::StatuArray>("drone_states", 10);
  target_traj_pub_ = nh.advertise<quadrotor_msgs::PolyTraj>("target_traj", 10);
  visPtr_ = std::make_shared<vis_utils::VisUtils>(nh);
  triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("triger", 10, triger_callback, ros::TransportHints().tcpNoDelay());

  ros::Timer cmd_timer = nh.createTimer(ros::Duration(0.01), cmdCallback);

  ros::Duration(1.0).sleep();

  ROS_WARN("[Traj server]: ready.");

  ros::spin();

  return 0;
}