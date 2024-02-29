#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nodelet/nodelet.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <traj_opt/traj_opt.h>
#include <quadrotor_msgs/PolyTraj.h>
#include <quadrotor_msgs/PolyTrajArray.h>
#include <quadrotor_msgs/StatuArray.h>
#include <traj_opt/dubins.h>

#include <Eigen/Core>
#include <atomic>
#include <thread>
#include <vis_utils/vis_utils.hpp>
#include <iostream>

namespace planning {

Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", ", ", "", "", " << ", ";");



class Nodelet : public nodelet::Nodelet {
 private:
  //ROS related
  std::thread initThread_;
  ros::Timer plan_timer_;

  ros::Subscriber triger_sub_;
  ros::Subscriber target_traj_sub_;
  ros::Subscriber swarm_state_sub_;

  ros::Publisher swarm_traj_pub_;

  std::shared_ptr<vis_utils::VisUtils> visPtr_;
  std::shared_ptr<traj_opt::TrajOpt> trajOptPtr_;

  double plan_duration_;

  std::atomic_bool triger_received_ = ATOMIC_VAR_INIT(false);

  // target trajectory
  bool receive_traj_ = false;
  Trajectory target_traj_poly_;
  ros::Time target_traj_start_time;

  // Parameters
  //fixwing swarm init params
  int drone_num_;
  int Piece_num_;  // inner point number
  Eigen::MatrixXd des_theta_;
  std::vector<double> des_theta_planar_;
  std::vector<double> head_theta_planar_;

  //Init target state
  Eigen::MatrixXd init_TargetState;
  Eigen::MatrixXd init_States_;

  std::vector<Trajectory> last_swarm_traj_;
  
  double rho_; // for curvature in Dubins search

  ros::Time Trigger_Time;

  //for replan
  Eigen::Vector3d tsignal;
  std::vector<Piece> ps;
  std::vector<Trajectory> Buffer_traj;
  double replan_ts;
  double close_dist_;
  double close_time_;

  double last_plan_deltaT, kDeltaT_;

  bool replan_; // replan signal

  static int printConfiguration(double q[3], double x, void* user_data) {
    printf("%f,%f,%f,%f\n", q[0], q[1], q[2], x);
    return 0;
  }

  void triger_callback(const geometry_msgs::PoseStampedConstPtr& msgPtr) {
    Trigger_Time = ros::Time::now();
    replan_ts = ros::Time::now().toSec();
    triger_received_ = true;
  }

  // Get and store the target trajectory  
  void TargetTrajCallback(const quadrotor_msgs::PolyTrajConstPtr &msgPtr) {
    target_traj_poly_.clear();
    target_traj_start_time = msgPtr->start_time;
    int piece_nums = msgPtr->duration.size();
    std::vector<double> dura(piece_nums);
    std::vector<CoefficientMat> cMats(piece_nums);
    for (int i = 0; i < piece_nums; ++i) {
      int i6 = i * 8;
      cMats[i].row(0) << msgPtr->coef_x[i6 + 0], msgPtr->coef_x[i6 + 1], msgPtr->coef_x[i6 + 2],
          msgPtr->coef_x[i6 + 3], msgPtr->coef_x[i6 + 4], msgPtr->coef_x[i6 + 5], msgPtr->coef_x[i6 + 6], msgPtr->coef_x[i6 + 7];
      cMats[i].row(1) << msgPtr->coef_y[i6 + 0], msgPtr->coef_y[i6 + 1], msgPtr->coef_y[i6 + 2],
          msgPtr->coef_y[i6 + 3], msgPtr->coef_y[i6 + 4], msgPtr->coef_y[i6 + 5], msgPtr->coef_y[i6 + 6], msgPtr->coef_y[i6 + 7];
      cMats[i].row(2) << msgPtr->coef_z[i6 + 0], msgPtr->coef_z[i6 + 1], msgPtr->coef_z[i6 + 2],
          msgPtr->coef_z[i6 + 3], msgPtr->coef_z[i6 + 4], msgPtr->coef_z[i6 + 5], msgPtr->coef_z[i6 + 6], msgPtr->coef_z[i6 + 7];

      dura[i] = msgPtr->duration[i];
    }
    target_traj_poly_.SetTraj(dura, cMats);
    receive_traj_ = true;

  }

  // Current swarm state receiver from simnode 
  void SwarmStateCallback(const quadrotor_msgs::StatuArrayPtr &msgPtr){
    assert(init_States_.cols()/4 == msgPtr->drone_status.size());
    for(int i = 0; i < drone_num_; i++){
        init_States_.col(i*4).x() = msgPtr->drone_status[i].position.x;
        init_States_.col(i*4).y() = msgPtr->drone_status[i].position.y;
        init_States_.col(i*4).z() = msgPtr->drone_status[i].position.z;

        init_States_.col(i*4+1).x() = msgPtr->drone_status[i].velocity.x;
        init_States_.col(i*4+1).y() = msgPtr->drone_status[i].velocity.y;
        init_States_.col(i*4+1).z() = msgPtr->drone_status[i].velocity.z;

        init_States_.col(i*4+2).x() = msgPtr->drone_status[i].acceleration.x;
        init_States_.col(i*4+2).y() = msgPtr->drone_status[i].acceleration.y;
        init_States_.col(i*4+2).z() = msgPtr->drone_status[i].acceleration.z;        
    }

  }

  // convert Trajectory to ROSMsgs
  void polyTraj2ROSMsg(quadrotor_msgs::PolyTraj &msg, Trajectory traj){
    Eigen::VectorXd durs = traj.getDurations();
    int piece_num = traj.getPieceNum();
    // std::cout<<"piece_num: "<<piece_num<<std::endl;
    // std::cout<<"durs: "<<durs.transpose()<<std::endl;
    msg.duration.resize(piece_num);
    msg.coef_x.resize(8 * piece_num);
    msg.coef_y.resize(8 * piece_num);
    msg.coef_z.resize(8 * piece_num);
    for (int i = 0; i < piece_num; ++i)
    {
      msg.duration[i] = durs(i);

      CoefficientMat cMat = traj.getPiece(i).getCoeffMat();
      int i8 = i * 8;
      for (int j = 0; j < 8; j++)
      {
        msg.coef_x[i8 + j] = cMat(0, j);
        msg.coef_y[i8 + j] = cMat(1, j);
        msg.coef_z[i8 + j] = cMat(2, j);
      }
    }
  }

  // evaluate if swarm is close to Target to decide whether stop replan
  bool isClose(){
    double tt = (ros::Time::now() - target_traj_start_time).toSec();
    Eigen::Vector3d Tp = target_traj_poly_.getPos(tt);
    double dis_max = 0;
    double dis_min = INFINITY;
    for(int i = 0; i<drone_num_; i++){
        Eigen::Vector3d pos = init_States_.col(i*4);
        double dist = (pos-Tp).norm();
        dis_max = dis_max < dist ? dist : dis_max;
        dis_min = dis_min > dist ? dist : dis_min;
    }
    std::cout<<"dis_max: "<<dis_max<<std::endl;
    std::cout<<"dis_min: "<<dis_min<<std::endl;
    double dis_mean = (dis_max+dis_min)/2;

    if(dis_mean < close_dist_){
        ROS_WARN("Replan_Stop Trigger!");
        return true;
    }
    return false;
  }

  bool ifStopReplan(){
    // all the planned traj time is the same
    if(last_swarm_traj_.empty()){
        return false;
    }

    ros::Time tnow = ros::Time::now();

    double t = last_swarm_traj_[0].getTotalDuration() - (tnow-last_swarm_traj_[0].start_time).toSec();
    ROS_WARN("Total Traj Time: %f. \n", t);
    if(t < close_time_){
        ROS_WARN("Replan_Stop Trigger!");
        return true;
    }
    return false;
  }


  // function for Dubins based initial path search
  void GetFeasiblePath(Eigen::MatrixXd initStates, double dt, std::vector<DubinsPath> & path, double& guessTime, double rho, double t){
    assert(initStates.cols() % 4 == 0);
    std::vector<Eigen::Vector3d> init_sp;

    // get init time
    Eigen::Vector3d init_tp = target_traj_poly_.getPos(t);
    double init_t = 0;
    for(int i =0; i < drone_num_; i++){
        init_sp.push_back(initStates.col(4*i));
        double tmp_t = (init_tp-initStates.col(4*i)).norm()/trajOptPtr_->vmax_;
        init_t = init_t < tmp_t ? tmp_t : init_t;
    }
    std::cout<<"init_time: "<<init_t<<std::endl;

    double dubins_length, dur = 20;
    double least_tc = INFINITY;
    Eigen::Vector3d guess_tp_;
    std::vector<DubinsPath> path_tmp(3);
    double tmp_cost;
    double tmp;
    double tail_for_dubins[3], head_for_dubins[3];
    for( double i = init_t; i < init_t+dur; i += dt ){

        // get time for dubins at guess time.
        tmp_cost = 0;
        guess_tp_ = target_traj_poly_.getPos(t+i);
        tail_for_dubins[0] = guess_tp_[0]; 
        tail_for_dubins[1] = guess_tp_[1];
        for(int j = 0; j < drone_num_; j++){
            tail_for_dubins[2] = des_theta_planar_[j];
            head_for_dubins[0] = init_sp[j].x();
            head_for_dubins[1] = init_sp[j].y();
            head_for_dubins[2] = head_theta_planar_[j];

            dubins_shortest_path(&path_tmp[j], head_for_dubins, tail_for_dubins, rho);

            dubins_length = dubins_path_length(&path_tmp[j]);
            tmp = dubins_length/trajOptPtr_->vmax_;

            tmp_cost += abs(tmp-i);
        }
        // std::cout<<"Time cost:"<< tmp_cost <<std::endl;

        if(tmp_cost < least_tc){
            least_tc = tmp_cost;
            for(int j = 0; j < drone_num_; j++){
                path[j] = path_tmp[j];
            }
            guessTime = i;
        }

        if(tmp_cost< 0.3){
            std::cout<<"early escape: "<<i<<std::endl;
            std::cout<<"Time cost:"<< tmp_cost <<std::endl;
            return;
        }
    }
    ROS_WARN("minimum time cost: %f", least_tc);
    // std::cout<<"minimum time cost:"<< least_tc <<std::endl;

    return;
  }

  // function for line based initial path search
  void GetFeasiblePath_compared(Eigen::MatrixXd initStates, double dt, double& guessTime, double t){
    assert(initStates.cols() % 4 == 0);
    std::vector<Eigen::Vector3d> init_sp;
    double v_mean = (trajOptPtr_->vmax_+trajOptPtr_->vmin_)/2;

    // get init time
    Eigen::Vector3d init_tp = target_traj_poly_.getPos(t);
    double init_t = 0;
    for(int i =0; i < drone_num_; i++){
        init_sp.push_back(initStates.col(4*i));
        double tmp_t = (init_tp-initStates.col(4*i)).norm()/v_mean;
        init_t = init_t < tmp_t ? tmp_t : init_t;
    }
    std::cout<<"init_time: "<<init_t<<std::endl;

    double dubins_length, dur = 20;
    double least_tc = INFINITY;
    Eigen::Vector3d guess_tp_;
    std::vector<DubinsPath> path_tmp(3);
    double time_cost = 0, tmp_cost = 0;
    for( double i = init_t; i < init_t+dur; i += dt ){

        // get time for dubins at guess time.
        guess_tp_ = target_traj_poly_.getPos(t+i);

        for(int j = 0; j < drone_num_; j++){

            dubins_length = (guess_tp_-initStates.col(j*4)).norm();
            double tmp = dubins_length/v_mean;

            tmp_cost += abs(tmp-i);

        }

        if(tmp_cost < least_tc){
            least_tc = tmp_cost;
            guessTime = i;
        }

        if(tmp_cost< 0.3){
            std::cout<<"early escape: "<<i<<std::endl;
            std::cout<<"Time cost:"<< tmp_cost <<std::endl;
            return;
        }

        tmp_cost = 0; 
    }
    std::cout<<"minimum time cost:"<< least_tc <<std::endl;

    return;
  }
  
  // Replan initialization 
  void replan_init(){
       ros::Time t_now = ros::Time::now();
        Buffer_traj.clear();
        Buffer_traj.resize(drone_num_);

        for(int i = 0; i < drone_num_; i++){
            double ts = (t_now - last_swarm_traj_[i].start_time).toSec();
            double tflag1, tflag2, tflag_tmp;
            tflag2 = ts + kDeltaT_ * last_plan_deltaT;

            if(tflag2 > last_swarm_traj_[i].getTotalDuration()){
                tflag2 = 0.66 * last_swarm_traj_[i].getTotalDuration();
            }

            init_States_.col(4*i) = last_swarm_traj_[i].getPos(tflag2);
            init_States_.col(4*i+1) = last_swarm_traj_[i].getVel(tflag2);
            init_States_.col(4*i+2) = last_swarm_traj_[i].getAcc(tflag2);
            init_States_.col(4*i+3) = last_swarm_traj_[i].getSnp(tflag2);

            tflag_tmp = tflag2;
            int cur_Idpiece = last_swarm_traj_[i].locatePieceIdx(ts);
            int replan_Idpiece = last_swarm_traj_[i].locatePieceIdx(tflag_tmp);

            Eigen::VectorXd duration_ = last_swarm_traj_[i].getDurations();

            tflag1 = 0;
            for(int j = 0; j < cur_Idpiece; j++) tflag1 += duration_[j];
            tsignal[i] = tflag1 + last_swarm_traj_[i].start_time.toSec();

            assert(cur_Idpiece<=replan_Idpiece);
            if(cur_Idpiece == replan_Idpiece){
                Piece ps = last_swarm_traj_[i].getPiece(cur_Idpiece);
                ps.setDuration(tflag2-tflag1);
                Buffer_traj[i].emplace_back(ps);
            }
            else{
                for( int idx=cur_Idpiece; idx<replan_Idpiece; idx++){
                    Piece ps = last_swarm_traj_[i].getPiece(idx);
                    Buffer_traj[i].emplace_back(ps);
                    tflag1 += duration_[idx];
                }
                Piece ps = last_swarm_traj_[i].getPiece(replan_Idpiece);
                ps.setDuration(tflag2-tflag1);
                Buffer_traj[i].emplace_back(ps);
            }
            replan_ts = last_swarm_traj_[i].start_time.toSec() + tflag2;      
        }
  }

  // main func for planning 
  void debug_timer_callback(const ros::TimerEvent& event) {
    auto tic = std::chrono::steady_clock::now();
    if (!triger_received_ || !receive_traj_ ) {
      return;
    }

    replan_ts = ros::Time::now().toSec(); 
    
    // set init states for replan
    // replanning must has the data of last planned trajs
    if(!last_swarm_traj_.empty() && last_plan_deltaT != -1)
        replan_init();

    //if close enough, stop replanning
    if(ifStopReplan()){
        triger_received_ = false;
        return;
    }


    //NOTE: Dubins based Path Search
    double t_exe = replan_ts - target_traj_start_time.toSec();
    bool generate_new_traj_success = false;
    std::vector<Trajectory> swarm_traj;
    swarm_traj.resize(drone_num_);

    for(int i = 0; i< drone_num_; i++){
        des_theta_planar_.push_back(atan(des_theta_.col(i).y()/(des_theta_.col(i).x()+1e-6)));
        head_theta_planar_.push_back(atan(init_States_.col(4*i+1)[1]/(init_States_.col(4*i+1)[0]+1e-4)));
    }

	std::vector<DubinsPath> DubinsOptimalPath;
    double dt = 0.1;
	double guess_t;

    DubinsOptimalPath.resize(3);

    auto tic1 = std::chrono::steady_clock::now();
	GetFeasiblePath(init_States_, dt, DubinsOptimalPath, guess_t, rho_, t_exe);
    auto toc1 = std::chrono::steady_clock::now();
    std::cout<<"front-end time cost: "<<(toc1-tic1).count() * 1e-6<<"ms"<<std::endl;

	ROS_INFO("Getting inner points"); 
    double dubins_length;
    Eigen::MatrixXd innerP(3, drone_num_ * (Piece_num_-1));
    for( int i = 0; i < drone_num_; i++){
        dubins_length = dubins_path_length(&DubinsOptimalPath[i]);
        for(int j=1; j < Piece_num_; j++){
            double x = dubins_length/Piece_num_ * j;
            double q1[3];
			double dz = (target_traj_poly_.getPos(t_exe + guess_t).z() - init_States_.col(i*4).z())/Piece_num_;
            dubins_path_sample(&DubinsOptimalPath[i], x, q1 );
						
            innerP.col(i * (Piece_num_-1) + j-1) = Eigen::Vector3d(q1[0],q1[1],init_States_.col(4 * i).z() + dz * j);
        }
    }

    double init_t = guess_t/Piece_num_;
    std::cout<<"init_t for every piece: "<<init_t<<std::endl;

    trajOptPtr_->setTargetTraj(target_traj_poly_);
    trajOptPtr_->setTargetTheta(des_theta_);
    trajOptPtr_->setExcue_T(t_exe);

    /* For Visualization*/
    // ROS_INFO("Initial trajectory visualization"); 
    // std::vector<Eigen::Vector3d> dubins_path, dubins_path_sampled;
    // std::string id_sample;
    // for( int i = 0; i < drone_num_; i++){
    //     dubins_path_sampled.push_back(init_States_.col(4 * i));
    //     for(int j=0; j<Piece_num_-1; j++)
    //         dubins_path_sampled.push_back(innerP.col(i * (Piece_num_-1) + j));
    //     dubins_path_sampled.push_back(target_traj_poly_.getPos(t_exe+guess_t));
    //     id_sample = "dubins_path_sampled_" + std::to_string(i);
    //     visPtr_->visualize_path(dubins_path_sampled, id_sample);

    //     dubins_length = dubins_path_length(&DubinsOptimalPath[i]);
    //     for(double t = 0.2; t <= dubins_length; t+=0.2){
    //         double q1[3];
	// 		double dz = (target_traj_poly_.getPos(t_exe+guess_t).z() - init_States_.col(i*4).z())/dubins_length;
    //         dubins_path_sample(&DubinsOptimalPath[i], t, q1 );
    //         dubins_path.push_back(Eigen::Vector3d(q1[0],q1[1], init_States_.col(i*4).z() + dz * t));
    // 		}
    //     id_sample = "dubins_path_" + std::to_string(i);
    //     visPtr_->visualize_path(dubins_path, id_sample);
    //     dubins_path.clear();
    //     dubins_path_sampled.clear();
    // }
    

    /* test for dubins front-end*/
    // double G_t;
    // tic1 = std::chrono::steady_clock::now();
	// GetFeasiblePath_compared(init_States_, dt, G_t, t_exe);
    // toc1 = std::chrono::steady_clock::now();
    // std::cout<<"front-end-test time cost: "<<(toc1-tic1).count() * 1e-6<<"ms"<<std::endl;
    
    // init_t = G_t/Piece_num_;

    // for(int i =0; i<drone_num_;i++){
    //     double dd = 1/double(Piece_num_);
    //     // std::cout<<"dd: "<<dd<<std::endl;
    //     Eigen::Vector3d dx = dd * (target_traj_poly_.getPos(G_t+t_exe)-init_States_.col(4*i));
    //     // std::cout<<"x1: "<<target_traj_poly_.getPos(G_t+t_exe).transpose()<<"x2: "<<init_States_.col(4*i).transpose()<<std::endl;
    //     // std::cout<<"dx: "<<dx.transpose()<<std::endl;
    //     for(int j=0; j<Piece_num_-1; j++){
    //         innerP.col(i*(Piece_num_-1)+j) = dx * (j+1) + init_States_.col(4*i);
    // }
    // }
    // std::cout<<"innerP: "<<innerP.transpose()<<std::endl;
    

    //NOTE: Centralized trajectory optimization
    ROS_INFO("Centralized trajectory optimization"); 
    generate_new_traj_success = trajOptPtr_->generate_traj(init_States_, innerP, swarm_traj, init_t);

    /* ----- Optimized Trajectory Visualization -------
    ROS_INFO("Optimized trajectory visualization");
    if (generate_new_traj_success) {
			double ddt = 0.01;
			for (double t = 0; t <= swarm_traj[0].getTotalDuration(); t += ddt) {
				visPtr_->visualize_traj(target_traj_poly_,"target_traj_zyh");

				Eigen::Vector3d target_p = target_traj_poly_.getPos(t);
				Eigen::Vector3d target_v = target_traj_poly_.getVel(t);

				visPtr_->visualize_arrow(target_p, target_p + 1.0 * target_v.normalized(), "target_dir", vis_utils::red);
				
				for(int i = 0 ; i < drone_num_; i++){
					
				    id_sample = "optimized_traj_" + std::to_string(i);
        	        visPtr_->visualize_traj(swarm_traj[i], id_sample);
        	// Eigen::Vector3d tail_pos = swarm_traj[i].getPos(swarm_traj[i].getTotalDuration());
        	// Eigen::Vector3d tail_vel = swarm_traj[i].getVel(swarm_traj[i].getTotalDuration());
        	// id_sample = "tail_vel_" + std::to_string(i);
        	// visPtr_->visualize_arrow(tail_pos, tail_pos + 1.0 * tail_vel.normalized(), id_sample);

				    Eigen::Vector3d p = swarm_traj[i].getPos(t);
					Eigen::Vector3d v = swarm_traj[i].getVel(t);
					id_sample = "drone_dir_" + std::to_string(i);
					visPtr_->visualize_arrow(p, p + 1.0 * v.normalized(), id_sample, vis_utils::yellow);
				}


			}
    }
    */

    if (!generate_new_traj_success) {
      triger_received_ = false;
      return;
    }

    if(last_swarm_traj_.empty())  last_swarm_traj_.resize(drone_num_);

    if(generate_new_traj_success){
        quadrotor_msgs::PolyTrajArray swarm_msg;
        ros::Time start_time = ros::Time::now();
        swarm_msg.header.frame_id = "world";
        swarm_msg.header.stamp.sec = start_time.toSec();
        for(int i = 0; i< drone_num_; i++){
            quadrotor_msgs::PolyTraj traj_msg;
            traj_msg.drone_id = i;
            traj_msg.order = 7;

            if(!Buffer_traj.empty()){
                traj_msg.start_time = ros::Time(tsignal[i]);
                Trajectory traj_tmp= Buffer_traj[i];
                traj_tmp.append(swarm_traj[i]);
                last_swarm_traj_[i] = traj_tmp;
                last_swarm_traj_[i].start_time = ros::Time(tsignal[i]);
                polyTraj2ROSMsg(traj_msg, traj_tmp);
            }
            else{
                last_swarm_traj_[i] = swarm_traj[i];
                last_swarm_traj_[i].start_time = start_time;
                
                traj_msg.start_time = start_time;
                polyTraj2ROSMsg(traj_msg, swarm_traj[i]);
            }
            swarm_msg.swarm_traj.push_back(traj_msg);
        }
        if(Buffer_traj.empty())  {
            Buffer_traj.resize(drone_num_);
        }
        swarm_traj_pub_.publish(swarm_msg);
    }
    auto toc = std::chrono::steady_clock::now();

    std::cout << "Total time costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;
    last_plan_deltaT = (toc - tic).count() * 1e-9;
    std::cout<<"last_plan_deltaT: "<<last_plan_deltaT<<std::endl;

    if(!replan_)
        triger_received_ = false;
  }

  void init(ros::NodeHandle& nh) {
    // set parameters of planning
	nh.getParam("drone_num", drone_num_);
	nh.getParam("Piece_num", Piece_num_);
    nh.getParam("plan_duration",plan_duration_);
	nh.getParam("rho", rho_);
    nh.getParam("close_dist", close_dist_);
    nh.getParam("close_time", close_time_);
    nh.param("kDeltaT", kDeltaT_, 1.8);
    nh.param("replan", replan_, false);

    //data initialization
    des_theta_.resize(3,drone_num_);
	init_States_.resize(3,4 * drone_num_);
	init_States_.setZero();

	std::string pos_str, vel_str, desTheta_str;
	for(int i = 0; i<drone_num_; i++){
		pos_str = "init_pos_" + std::to_string(i);
		nh.getParam(pos_str+"_x", init_States_.col(4*i).x());
		nh.getParam(pos_str+"_y", init_States_.col(4*i).y());
		nh.getParam(pos_str+"_z", init_States_.col(4*i).z());

		vel_str = "init_vel_" + std::to_string(i);
		nh.getParam(vel_str+"_x", init_States_.col(4*i+1).x());
		nh.getParam(vel_str+"_y", init_States_.col(4*i+1).y());
		nh.getParam(vel_str+"_z", init_States_.col(4*i+1).z());

		desTheta_str = "des_theta_"	+ std::to_string(i);
		nh.getParam(desTheta_str + "_x", des_theta_.col(i).x());
		nh.getParam(desTheta_str + "_y", des_theta_.col(i).y());
		nh.getParam(desTheta_str + "_z", des_theta_.col(i).z());		
	}

    last_plan_deltaT = -1;

    visPtr_ = std::make_shared<vis_utils::VisUtils>(nh);
    trajOptPtr_ = std::make_shared<traj_opt::TrajOpt>(nh);

    plan_timer_ = nh.createTimer(ros::Duration(plan_duration_), &Nodelet::debug_timer_callback, this);

    triger_sub_ = nh.subscribe<geometry_msgs::PoseStamped>("triger", 10, &Nodelet::triger_callback, this, ros::TransportHints().tcpNoDelay());
    swarm_state_sub_ = nh.subscribe("drone_states", 10, &Nodelet::SwarmStateCallback, this, ros::TransportHints().tcpNoDelay());
    target_traj_sub_ = nh.subscribe<quadrotor_msgs::PolyTraj>("target_traj", 10, &Nodelet::TargetTrajCallback, this, ros::TransportHints().tcpNoDelay());
    
    swarm_traj_pub_ = nh.advertise<quadrotor_msgs::PolyTrajArray>("Swarm_trajs", 10);

    ROS_WARN("Planning node initialized!");
  }

 public:
  void onInit(void) {
    ros::NodeHandle nh(getMTPrivateNodeHandle());
    initThread_ = std::thread(std::bind(&Nodelet::init, this, nh));
  }
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

}  // namespace planning

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(planning::Nodelet, nodelet::Nodelet);