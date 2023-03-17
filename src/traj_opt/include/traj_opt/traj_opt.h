#pragma once
#include <ros/ros.h>

#include <chrono>
#include <thread>
#include <vis_utils/vis_utils.hpp>

#include "minco_raw.hpp"

namespace traj_opt {

struct LocalTrajData
  {
    Trajectory traj;
    int drone_id; // A negative value indicates no received trajectories.
    int traj_id;
    double duration;
    double start_time; // world time
    double end_time;   // world time
    Eigen::Vector3d start_pos;
};

typedef std::vector<LocalTrajData> SwarmTrajData;

class TrajOpt {
 public:
  ros::NodeHandle nh_;
  std::shared_ptr<vis_utils::VisUtils> visPtr_;
  bool pause_debug_ = false;
  // # pieces and # key points
  int N_, K_, dim_t_, dim_p_;
  // weight for time regularization term
  double rhoT_;

  double vmax_, amax_, vmin_, amin_;
  double dSwarmMin_;
  double omegamax_, Curmax_;
  double vmean_, vgap_, amean_, agap_;
  double v_sqr_mean_, v_sqr_gap_;
  double rhoP_, rhoV_, rhoA_, rhoVtail_, rhoPswarm_, rhoC_;
  // landing parameters
  double v_plus_, robot_l_, robot_r_, platform_r_;
  // MINCO Optimizer
  minco::MINCO_S4_Uniform mincoOpt_, debugOpt;
  std::vector<minco::MINCO_S4_Uniform> swarm_mincoOpt_, swarm_debugOpt_;
  std::vector<Eigen::MatrixXd> initS_, initE_;
  // duration of each piece of the trajectory
  Eigen::VectorXd t_;
  double* x_;

  bool monitorUse_;
  int pausems_;

  double swarm_gdT;
  double excue_t;

  std::vector<Eigen::Vector3d> tracking_ps_;
  std::vector<Eigen::Vector3d> tracking_visible_ps_;
  std::vector<double> tracking_thetas_;

  // forward T
  Eigen::VectorXd times;
  int dim_tail_;

  // target trajectory
  Trajectory target_traj_poly_;

  // target direction
  std::vector<Eigen::Vector3d>des_theta;
//   Eigen::Vector3d des_theta1_;

  int drone_num_;

 public:
  TrajOpt(ros::NodeHandle& nh);
  ~TrajOpt() {}

  int optimize(const double& delta = 1e-4);

  bool generate_traj(const Eigen::MatrixXd& initState,
                    const Eigen::MatrixXd& innerPts,
                    std::vector<Trajectory>& traj,
                    const double& init_t);

  void addTimeIntPenalty(double& cost);

  void addTimeIntPenalty_Swarm(double& cost);

  Eigen::VectorXd forwardT(const Eigen::VectorXd& t);

  Eigen::VectorXd backwardT(const Eigen::VectorXd& T);

  bool grad_cost_v(const Eigen::Vector3d& v,
                   Eigen::Vector3d& gradv,
                   double& costv);

  bool grad_cost_a(const Eigen::Vector3d& a,
                   Eigen::Vector3d& grada,
                   double& costa);

  bool grad_cost_a(const Eigen::Vector3d& a,
                   const Eigen::Vector3d& v,
                   Eigen::Vector3d& grada,
                   Eigen::Vector3d& gradv,
                   double& costa);

  bool grad_curvature_check(const Eigen::Vector3d& a,
                            const Eigen::Vector3d& v,
                            Eigen::Vector3d& grada,
                            Eigen::Vector3d& gradv,
                            double& cost);

  Eigen::VectorXd addLayerTGrad(const Eigen::VectorXd& t,
                                const Eigen::VectorXd& gradT);

  void setTargetTraj(Trajectory& target_traj);

  void setTargetTheta(Eigen::MatrixXd target_theta);

  bool grad_cost_dirct(const Eigen::Vector3d& v_dir,
                       Eigen::Vector3d& grad_vdir,
                       double& cost_vdir);
    
  bool grad_cost_limitv(const double& v,
                        double& gradv,
                        double& costv);
  bool grad_collision_check(const Eigen::Vector3d& p1,
							const Eigen::Vector3d& p2,
                          	Eigen::Vector3d& gradp1,
							Eigen::Vector3d& gradp2,
                          	double& costp); 

  void setExcue_T(double& t); 
};

}  // namespace traj_opt