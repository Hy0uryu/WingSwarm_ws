#include <traj_opt/traj_opt.h>

#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt {

static Eigen::Vector3d car_p_, car_v_;
static Eigen::Vector3d tail_q_v_;
static Eigen::Vector3d g_(0, 0, -9.8);
static Eigen::Vector3d land_v_;
static Eigen::Vector3d v_t_x_, v_t_y_;
static Trajectory init_traj_;
static double init_tail_f_;
static Eigen::Vector2d init_vt_;
static bool initial_guess_ = false;

static double thrust_middle_, thrust_half_;

static double tictoc_innerloop_;
static double tictoc_integral_;

static int iter_times_ = 0;

static bool q2v(const Eigen::Quaterniond& q,
                Eigen::Vector3d& v) {
  Eigen::MatrixXd R = q.toRotationMatrix();
  v = R.col(2);
  return true;
}
static Eigen::Vector3d f_N(const Eigen::Vector3d& x) {
  return x.normalized();
}
static Eigen::MatrixXd f_DN(const Eigen::Vector3d& x) {
  double x_norm_2 = x.squaredNorm();
  return (Eigen::MatrixXd::Identity(3, 3) - x * x.transpose() / x_norm_2) / sqrt(x_norm_2);
}
static Eigen::MatrixXd f_D2N(const Eigen::Vector3d& x, const Eigen::Vector3d& y) {
  double x_norm_2 = x.squaredNorm();
  double x_norm_3 = x_norm_2 * x.norm();
  Eigen::MatrixXd A = (3 * x * x.transpose() / x_norm_2 - Eigen::MatrixXd::Identity(3, 3));
  return (A * y * x.transpose() - x * y.transpose() - x.dot(y) * Eigen::MatrixXd::Identity(3, 3)) / x_norm_3;
}

// SECTION  variables transformation and gradient transmission
static double smoothedL1(const double& x,
                         double& grad) {
  static double mu = 0.03;
  if (x < 0.0) {
    return 0.0;
  } else if (x > mu) {
    grad = 1.0;
    return x - 0.5 * mu;
  } else {
    const double xdmu = x / mu;
    const double sqrxdmu = xdmu * xdmu;
    const double mumxd2 = mu - 0.5 * x;
    grad = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
    return mumxd2 * sqrxdmu * xdmu;
  }
}

static double penF(const double& x, double& grad) {
  static double eps = 0.05;
  static double eps2 = eps * eps;
  static double eps3 = eps * eps2;
  if (x < 2 * eps) {
    double x2 = x * x;
    double x3 = x * x2;
    double x4 = x2 * x2;
    grad = 12 / eps2 * x2 - 4 / eps3 * x3;
    return 4 / eps2 * x3 - x4 / eps3;
  } else {
    grad = 16;
    return 16 * (x - eps);
  }
}

static double smoothedTriple(const double& x,
                         double& grad) {
  if (x < 0.0) {
    return 0.0;
  } 
  else {
    grad = 3 * x * x;
    return x * x * x;
  }
}

static double smoothed01(const double& x,
                         double& grad) {
  static double mu = 0.01;
  static double mu4 = mu * mu * mu * mu;
  static double mu4_1 = 1.0 / mu4;
  if (x < -mu) {
    grad = 0;
    return 0;
  } else if (x < 0) {
    double y = x + mu;
    double y2 = y * y;
    grad = y2 * (mu - 2 * x) * mu4_1;
    return 0.5 * y2 * y * (mu - x) * mu4_1;
  } else if (x < mu) {
    double y = x - mu;
    double y2 = y * y;
    grad = y2 * (mu + 2 * x) * mu4_1;
    return 0.5 * y2 * y * (mu + x) * mu4_1 + 1;
  } else {
    grad = 0;
    return 1;
  }
}

static Eigen::MatrixXd skewMatrix(const Eigen::Vector3d& v) {
  Eigen::MatrixXd A(3, 3);
  A << 0, -v.z(), v.y(),
      v.z(), 0, -v.x(),
      -v.y(), v.x(), 0;
  return A;
}

static double expC2(double t) {
  return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
                 : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
}
static double logC2(double T) {
  return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
}
static inline double gdT2t(double t) {
  if (t > 0) {
    return t + 1.0;
  } else {
    double denSqrt = (0.5 * t - 1.0) * t + 1.0;
    return (1.0 - t) / (denSqrt * denSqrt);
  }
}

Eigen::VectorXd TrajOpt::forwardT(const Eigen::VectorXd& t) {
  assert(t.size() == N_);
  Eigen::VectorXd T(N_);
  for (int i = 0; i < N_; ++i) {
    T(i) = expC2(t(i));
  }
  return T;
}

Eigen::VectorXd TrajOpt::backwardT(const Eigen::VectorXd& T) {
  assert(T.size() == N_);
  Eigen::VectorXd t(N_);
  for (int i = 0; i < N_; ++i) {
    t(i) = logC2(T(i));
  }
  return t;
}

Eigen::VectorXd TrajOpt::addLayerTGrad(const Eigen::VectorXd& t,
                                       const Eigen::VectorXd& gradT) {
  assert(t.size() == N_);
  Eigen::VectorXd gradt(N_);
  for (int i = 0; i < N_; ++i) {
    gradt(i) = gradT(i) * gdT2t(t(i));
  }
  return gradt;
}

static inline double objectiveFunc_WingSwarm_(void* ptrObj,
                                   const double* x,
                                   double* grad,
                                   const int n) {
  iter_times_++;
//   std::cout<<"iteration: "<< iter_times_ <<std::endl;
  TrajOpt& obj = *(TrajOpt*)ptrObj;
  const double& x_t = x[0]; 

  Eigen::Map<const Eigen::MatrixXd>x_p(x + obj.dim_t_, 3, obj.drone_num_ * obj.dim_p_); 
  // Eigen::Map<Eigen::MatrixXd> v_tail(x_ + dim_t_ + 3 * dim_p_, 3, dim_tail_ );
  Eigen::Map<const Eigen::VectorXd> v_tail(x + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_, obj.dim_tail_);

  double& gradt = grad[0];
  Eigen::Map<Eigen::MatrixXd> gradp(grad + obj.dim_t_, 3, obj.drone_num_ * obj.dim_p_);
  Eigen::Map<Eigen::VectorXd> gradvtail(grad + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_, obj.dim_tail_);

  double dT = expC2(x_t);
//   std::cout<<"dT: "<<dT<<std::endl;
  double end_time = dT * obj.N_ + obj.excue_t;
  
//   auto tic = std::chrono::steady_clock::now();
//   auto toc = std::chrono::steady_clock::now();
//   tictoc_innerloop_ += (toc - tic).count();
  double cost = 0;
  obj.swarm_gdT = 0;
//   std::cout<<"Set the tail state and minco generation"<<std::endl;
  for(int i=0; i<obj.drone_num_; i++){

    // set the tail state
    obj.initE_[i].col(0) = obj.target_traj_poly_.getPos(end_time);
    obj.initE_[i].col(1) = v_tail[i] * obj.des_theta[i];
    // obj.initE_[i].col(0) = Eigen::Vector3d(3,3,5);
    // obj.initE_[i].col(1).setZero();
    obj.initE_[i].col(2).setZero();
    obj.initE_[i].col(3).setZero();

    obj.swarm_mincoOpt_[i].generate(obj.initS_[i], obj.initE_[i], x_p.block(0, i * obj.dim_p_ , 3, obj.dim_p_), dT);
    cost += obj.swarm_mincoOpt_[i].getTrajSnapCost();
    obj.swarm_mincoOpt_[i].calGrads_CT();

  }

    obj.addTimeIntPenalty_Swarm(cost);

    gradt = 0;
    gradvtail.setZero();
//   std::cout<<"set the time penalty and tail penalty"<<std::endl;
  for(int i=0; i<obj.drone_num_; i++){
    obj.swarm_mincoOpt_[i].calGrads_PT();
    gradp.block(0, i * obj.dim_p_ , 3, obj.dim_p_) = obj.swarm_mincoOpt_[i].gdP;

    double PatialrhoToTime = 0;
    PatialrhoToTime += obj.swarm_mincoOpt_[i].gdTail.col(0).dot(obj.N_ * obj.target_traj_poly_.getVel(end_time)) ;
    obj.swarm_mincoOpt_[i].gdT += PatialrhoToTime;
    
    gradvtail(i) += obj.swarm_mincoOpt_[i].gdTail.col(1).dot(obj.des_theta[i]);
    double grad_tmp, cost_vlimit;
    if(obj.grad_cost_limitv(v_tail(i),grad_tmp, cost_vlimit)){
        gradvtail(i) += grad_tmp;
        cost += cost_vlimit;
    }

    gradt += obj.swarm_mincoOpt_[i].gdT * gdT2t(x_t);
  }
  
  gradt += obj.swarm_gdT * gdT2t(x_t);

  gradt += obj.rhoT_ * gdT2t(x_t) * obj.N_;
  cost  += obj.rhoT_ * dT * obj.N_;

//   std::cout<<"swarm_gdT: "<< obj.swarm_gdT<<std::endl;
//   std::cout<<"gdt: "<<gradt<<std::endl;

//   for(int i = 1; i< 3 * obj.drone_num_ * obj.dim_p_; i++) std::cout<<grad[i]<<" ";
//   std::cout<<std::endl;

//   std::cout<<"gradVtail: "<<gradvtail.transpose()<<std::endl;
//   std::cout<<"cost: "<<cost<<std::endl;


  return cost;
}

static inline int processMonitor(void* ptrObj,
                            const double* x,
                            const double* grad,
                            const double fx,
                            const double xnorm,
                            const double gnorm,
                            const double step,
                            int n,
                            int k,
                            int ls) {
    TrajOpt& obj = *(TrajOpt*)ptrObj;
    if (obj.monitorUse_) {
        const double x_t=x[0];
        Eigen::Map<const Eigen::MatrixXd> x_p(x + obj.dim_t_, 3, obj.dim_p_);
        Eigen::Map<const Eigen::VectorXd> v_tail(x + obj.dim_t_ + 3 * obj.dim_p_, 3);

        double dT = expC2(x_t);
    
        double end_time = dT * obj.N_;

        Eigen::MatrixXd tail_tmp;
        tail_tmp.resize(3,4);

        // tail_tmp.col(0) = obj.target_traj_poly_.getPos(end_time);
        // tail_tmp.col(1) = v_tail(0) * obj.des_theta1_;
        // tail_tmp.col(2).setZero();
        // tail_tmp.col(3).setZero();

        // obj.debugOpt.generate(obj.initS_, tail_tmp, x_p, times);
        obj.debugOpt.generate(obj.initS_[0], obj.initE_[0], x_p, dT);
        obj.visPtr_->visualize_traj(obj.debugOpt.getTraj(),"optimizing_traj");

        // NOTE pause
        std::this_thread::sleep_for(std::chrono::milliseconds(obj.pausems_));
    }
  return 0;
}

bool TrajOpt::generate_traj(const Eigen::MatrixXd& initState,
                    const Eigen::MatrixXd& innerPts,
                    std::vector<Trajectory>& swarm_traj_,
                    const double& init_t){
    assert(initState.cols() % 4 == 0);
    drone_num_ = initState.cols()/4;
    N_ = innerPts.cols()/drone_num_ + 1;
    dim_t_ = 1;
    dim_p_ = N_ - 1;
    dim_tail_ = drone_num_;
    assert(N_ > 1);
    std::cout<<"innerPts_num_:"<<innerPts.cols()<<std::endl;

	x_ = new double[dim_t_ + 3 * drone_num_ * dim_p_ + dim_tail_];
	double& x_t = x_[0];

    Eigen::Map<Eigen::MatrixXd>innerp_set(x_ + dim_t_, 3, drone_num_ * dim_p_);
    Eigen::Map<Eigen::VectorXd> v_tail(x_ + dim_t_ + 3 * drone_num_ * dim_p_, dim_tail_);

    innerp_set = innerPts;

    swarm_mincoOpt_.resize(drone_num_);
    swarm_debugOpt_.resize(drone_num_);
    initS_.resize(drone_num_);
    initE_.resize(drone_num_);
    
    for(int i = 0; i < drone_num_; i++){
        swarm_mincoOpt_[i].reset(N_);
        swarm_debugOpt_[i].reset(N_);
        initS_[i].resize(3, 4);
        initE_[i].resize(3, 4);
        initS_[i] = initState.block<3,4>(0, i * 4);
    }

    x_t = logC2(init_t);
    std::cout<<"x_t:"<<x_t<<std::endl;
    v_tail.fill(vmax_);

    // initE_[0].col(0) = Eigen::Vector3d(3,3,5);
    // for(int i =1; i < N_; i++ ){
    //     Eigen::Vector3d ds = (initE_[0].col(0)-initS_[0].col(0))/N_;
    //     innerp_set.col(i-1) = ds * i + initS_[0].col(0);
    // }

    // double ds = (initE_[0].col(0)-initS_[0].col(0)).norm()/dim_p_;
    // x_t = logC2(ds/vmin_);

    // std::cout<<"optimization start!"<<std::endl;
    // std::cout<<"total num: "<<drone_num_<<std::endl;
    // for(int i =0; i< drone_num_; i++){
    //     std::cout<<"drone_num: ["<<i<<"]"<<std::endl;
    //     std::cout<<"INIT STATE:"<<std::endl;
    //     std::cout<<initS_[i].transpose()<<std::endl;
    //     std::cout<<"INTERIOR POINTS:"<<std::endl;
    //     std::cout<<innerp_set.block(0, i * dim_p_ , 3, dim_p_).transpose()<<std::endl;
    // }
    // std::cout<<"init_T: "<<ds/vmax_<<std::endl;

    // for(int i = 0; i<dim_t_ + 3 * drone_num_ * dim_p_ + dim_tail_; i++ )
    //     std::cout<<x_[i]<<" ";
    

    // NOTE optimization
    lbfgs::lbfgs_parameter_t lbfgs_params;
    lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
    // lbfgs_params.mem_size = 256;  //64
    // lbfgs_params.past = 0;
    // lbfgs_params.delta = 1e-8;
    // lbfgs_params.g_epsilon = 0.1;
    // lbfgs_params.min_step = 1e-128;
    // lbfgs_params.line_search_type = 0;  //0

    lbfgs_params.mem_size = 128;  //64
    lbfgs_params.past = 3;
    lbfgs_params.delta = 1e-6;
    lbfgs_params.g_epsilon = 0.01;
    lbfgs_params.min_step = 1e-64;
    lbfgs_params.line_search_type = 0;  //0

    /* perching parma
    lbfgs_params.past = 3;
    lbfgs_params.g_epsilon = 0.0;
    lbfgs_params.min_step = 1e-16;
    lbfgs_params.delta = 1e-4;
    lbfgs_params.line_search_type = 0;
    */
    double minObjective;

    int opt_ret = 0;

    std::cout << "******************************" << std::endl;

    auto tic = std::chrono::steady_clock::now();

    //   std::cout<<"optimization start!"<<std::endl;
    opt_ret = lbfgs::lbfgs_optimize(dim_t_ + 3 * drone_num_ * dim_p_ + dim_tail_, x_, &minObjective,
                                    &objectiveFunc_WingSwarm_, nullptr,
                                    &processMonitor, this, &lbfgs_params);
    //   std::cout<<"optimization finish!"<<std::endl;  

    auto toc = std::chrono::steady_clock::now();

    std::cout << "\033[32m>ret: " << opt_ret << "\033[0m" << std::endl;

    std::cout << "optmization costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;
    if (pause_debug_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    if (opt_ret < 0) {
        delete[] x_;
        return false;
    }

    for(int i = 0; i < swarm_traj_.size(); i++ )
        swarm_traj_[i] = swarm_mincoOpt_[i].getTraj();

    delete[] x_;
    return true;
                    }

static void bvp(const double& t,
                const Eigen::MatrixXd i_state,
                const Eigen::MatrixXd f_state,
                CoefficientMat& coeffMat) {
  double t1 = t;
  double t2 = t1 * t1;
  double t3 = t2 * t1;
  double t4 = t2 * t2;
  double t5 = t3 * t2;
  double t6 = t3 * t3;
  double t7 = t4 * t3;
  CoefficientMat boundCond;
  boundCond.leftCols(4) = i_state;
  boundCond.rightCols(4) = f_state;

  coeffMat.col(0) = (boundCond.col(7) / 6.0 + boundCond.col(3) / 6.0) * t3 +
                    (-2.0 * boundCond.col(6) + 2.0 * boundCond.col(2)) * t2 +
                    (10.0 * boundCond.col(5) + 10.0 * boundCond.col(1)) * t1 +
                    (-20.0 * boundCond.col(4) + 20.0 * boundCond.col(0));
  coeffMat.col(1) = (-0.5 * boundCond.col(7) - boundCond.col(3) / 1.5) * t3 +
                    (6.5 * boundCond.col(6) - 7.5 * boundCond.col(2)) * t2 +
                    (-34.0 * boundCond.col(5) - 36.0 * boundCond.col(1)) * t1 +
                    (70.0 * boundCond.col(4) - 70.0 * boundCond.col(0));
  coeffMat.col(2) = (0.5 * boundCond.col(7) + boundCond.col(3)) * t3 +
                    (-7.0 * boundCond.col(6) + 10.0 * boundCond.col(2)) * t2 +
                    (39.0 * boundCond.col(5) + 45.0 * boundCond.col(1)) * t1 +
                    (-84.0 * boundCond.col(4) + 84.0 * boundCond.col(0));
  coeffMat.col(3) = (-boundCond.col(7) / 6.0 - boundCond.col(3) / 1.5) * t3 +
                    (2.5 * boundCond.col(6) - 5.0 * boundCond.col(2)) * t2 +
                    (-15.0 * boundCond.col(5) - 20.0 * boundCond.col(1)) * t1 +
                    (35.0 * boundCond.col(4) - 35.0 * boundCond.col(0));
  coeffMat.col(4) = boundCond.col(3) / 6.0;
  coeffMat.col(5) = boundCond.col(2) / 2.0;
  coeffMat.col(6) = boundCond.col(1);
  coeffMat.col(7) = boundCond.col(0);

  coeffMat.col(0) = coeffMat.col(0) / t7;
  coeffMat.col(1) = coeffMat.col(1) / t6;
  coeffMat.col(2) = coeffMat.col(2) / t5;
  coeffMat.col(3) = coeffMat.col(3) / t4;
}

void TrajOpt::addTimeIntPenalty_Swarm(double& cost) {
  Eigen::MatrixXd pos, vel, acc, jer, snp;
  Eigen::MatrixXd grad_p, grad_v, grad_a, grad_j;
  Eigen::Vector3d grad_tmp;
  double cost_tmp;
  Eigen::VectorXd cost_inner;
  Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
  double s1, s2, s3, s4, s5, s6, s7;
  double step, alpha;
  std::vector<Eigen::Matrix<double, 8, 3>> gradViola_c;
  double gradViola_t;
  double omg;                                                                
  pos.resize(3,drone_num_);
  vel.resize(3,drone_num_);
  acc.resize(3,drone_num_);
  jer.resize(3,drone_num_);
  snp.resize(3,drone_num_);
  grad_p.resize(3,drone_num_);
  grad_v.resize(3,drone_num_);
  grad_a.resize(3,drone_num_);
  grad_j.resize(3,drone_num_);
  cost_inner.resize(drone_num_);
  gradViola_c.resize(drone_num_);

  int innerLoop = K_ + 1;
  step = swarm_mincoOpt_[0].t(1) / K_;

  s1 = 0.0;
  for (int j = 0; j < innerLoop; ++j) {
    s2 = s1 * s1;
    s3 = s2 * s1;
    s4 = s2 * s2;
    s5 = s4 * s1;
    s6 = s4 * s2;
    s7 = s4 * s3;
    beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
    beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
    beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;
    beta3 << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120.0 * s3, 210.0 * s4;
    beta4 << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360.0 * s2, 840.0 * s3;
    alpha = 1.0 / K_ * j;
    omg = (j == 0 || j == innerLoop - 1) ? 0.5 : 1.0;

    for (int i = 0; i < N_; ++i) {
      std::vector<Eigen::Matrix<double,8,3>> c(drone_num_);
      for(int k = 0; k<drone_num_; k++){
        c[k] = swarm_mincoOpt_[k].c.block<8, 3>(i * 8, 0);
        pos.col(k) = c[k].transpose() * beta0;
        vel.col(k) = c[k].transpose() * beta1;
        acc.col(k) = c[k].transpose() * beta2;
        jer.col(k) = c[k].transpose() * beta3;
        snp.col(k) = c[k].transpose() * beta4;
      }

      grad_p.setZero();
      grad_v.setZero();
      grad_a.setZero();
      grad_j.setZero();
	  cost_inner.setZero();

	  // penalty of single drone
      for(int k = 0; k<drone_num_; k++){
				if(true){
					if (grad_cost_v(vel.col(k), grad_tmp, cost_tmp)) {
							grad_v.col(k) += grad_tmp;
							cost_inner(k) += cost_tmp;
					}
                    
                    Eigen::Vector3d grad_tmp1, grad_tmp2;
					if (grad_cost_a(acc.col(k), vel.col(k), grad_tmp1, grad_tmp2, cost_tmp)) {
                            grad_v.col(k) += grad_tmp2;
							grad_a.col(k) += grad_tmp1;
							cost_inner(k) += cost_tmp;
					}

                    if(grad_curvature_check(acc.col(k), vel.col(k), grad_tmp1, grad_tmp2, cost_tmp)){
                            // std::cout<<cost_tmp<<" "<<grad_tmp1.transpose()<<" "<<grad_tmp2.transpose()<<std::endl;
                            grad_v.col(k) += grad_tmp2;
							grad_a.col(k) += grad_tmp1;
							cost_inner(k) += cost_tmp;                       
                    }

                    // if (grad_cost_a(acc.col(k), grad_tmp, cost_tmp)) {
					// 		grad_a.col(k) += grad_tmp;
					// 		cost_inner(k) += cost_tmp;
					// }
      	}

      }

			// penalty among the drones
	  if(i<N_-1){
				Eigen::Vector3d grad_tmp1, grad_tmp2;
				double cost_swarm;
				for(int x = 0; x < drone_num_; x++)
						for( int y = x+1; y < drone_num_; y++){
                                // std::cout<<"x: "<<x<<" y: "<<y<<std::endl;
								if(grad_collision_check(pos.col(x),pos.col(y),grad_tmp1,grad_tmp2,cost_swarm)){
                                    // std::cout<<"cost: "<<cost_swarm<<std::endl;
                                    // std::cout<<"grad_tmp1: "<<grad_tmp1.transpose()<<std::endl;
                                    // std::cout<<"grad_tmp2: "<<grad_tmp2.transpose()<<std::endl;
									grad_p.col(x) += grad_tmp1;
									grad_p.col(y) += grad_tmp2;
									swarm_gdT += omg * cost_swarm / K_;
									cost += omg * step * cost_swarm;
								}
		}
	  }

      for(int k = 0; k<drone_num_; k++){
		gradViola_c[k] = beta0 * grad_p.col(k).transpose();
		gradViola_t = grad_p.col(k).transpose() * vel.col(k);
		gradViola_c[k] += beta1 * grad_v.col(k).transpose();
		gradViola_t += grad_v.col(k).transpose() * acc.col(k);
		gradViola_c[k] += beta2 * grad_a.col(k).transpose();
		gradViola_t += grad_a.col(k).transpose() * jer.col(k);
		gradViola_c[k] += beta3 * grad_j.col(k).transpose();
		gradViola_t += grad_j.col(k).transpose() * snp.col(k);

		swarm_mincoOpt_[k].gdC.block<8, 3>(i * 8, 0) += omg * step * gradViola_c[k];
		swarm_mincoOpt_[k].gdT += omg * (cost_inner(k) / K_ + alpha * step * gradViola_t);
		cost += omg * step * cost_inner(k);
      }						
    }
    s1 += step;
  }
}

bool TrajOpt::grad_cost_v(const Eigen::Vector3d& v,
                          Eigen::Vector3d& gradv,
                          double& costv) {
  /*- shit - */
  double vpen = (v.squaredNorm() - v_sqr_mean_) * (v.squaredNorm() - v_sqr_mean_) - v_sqr_gap_ * v_sqr_gap_;
  if (vpen > 0) {
    double grad = 0;
    costv = rhoV_ * smoothedTriple(vpen, grad);
    // costv = rhoV_ * smoothedL1(vpen, grad);
    gradv = rhoV_ * grad * 4 * (v.squaredNorm() - v_sqr_mean_) * v;
    return true;
  }
  return false;
  

//   double pen;
//   pen = v.squaredNorm() - vmax_ * vmax_;
//   if(pen > 0){
//     double grad = 0;
//     costv = rhoV_ * penF(pen, grad);
//     gradv = rhoV_ * grad * 2 * v;
//     return true;
//   }
//   else{
//     pen = vmin_ * vmin_ - v.squaredNorm() ;
//     if(pen > 0){
//         double grad = 0;
//         costv = rhoV_ * pen * pen * pen;
//         gradv = - 6 * rhoV_ * pen * pen * v;
//         return true;
//     }
//   }
//   return false;

}

bool TrajOpt::grad_cost_a(const Eigen::Vector3d& a,
                          Eigen::Vector3d& grada,
                          double& costa) {
  double apen = a.squaredNorm() - amax_ * amax_;
  if (apen > 0) {
    double grad = 0;
    costa = rhoA_ * smoothedTriple(apen, grad);
    grada = rhoA_ * grad * 2 * a;
    return true;
  }
  return false;
}

bool TrajOpt::grad_cost_a(const Eigen::Vector3d& a,
                          const Eigen::Vector3d& v,
                          Eigen::Vector3d& grada,
                          Eigen::Vector3d& gradv,
                          double& costa) {
  double apen;
  apen = v.normalized().dot(a) - vmax_;
  double grad = 0;
  if(apen > 0){
    costa = rhoA_ * smoothedTriple(apen, grad);
    gradv = rhoA_ * grad * f_DN(v) * a;
    grada = rhoA_ * grad * v.normalized();
  }
  else{
    apen = vmin_ - v.normalized().dot(a);
  if(apen > 0){
    costa = rhoA_ * smoothedTriple(apen, grad);
    gradv = - rhoA_ * grad * f_DN(v) * a;
    grada = - rhoA_ * grad * v.normalized();
  }
  }
  return false;
}

bool TrajOpt::grad_curvature_check(const Eigen::Vector3d& a,
                                   const Eigen::Vector3d& v,
                                   Eigen::Vector3d& grada,
                                   Eigen::Vector3d& gradv,
                                   double& cost) {
  double pen = (v.cross(a)).norm()/pow(v.norm(),3) - Curmax_;
//   std::cout<<"Cpen: "<<pen<<std::endl;
  if (pen > 0) {
    double grad = 0;
    cost = rhoC_ * smoothedTriple(pen, grad);
    Eigen::Vector3d Partialv = - (1/((v.cross(a)).norm()*pow(v.norm(),3))) * skewMatrix(a).transpose() * (v.cross(a)) - 3*(v.cross(a)).norm()/pow(v.norm(),5) * v;
    Eigen::Vector3d Partiala = (1 / (pow(v.norm(),3) * (v.cross(a)).norm())) * skewMatrix(v).transpose() * (v.cross(a));
    gradv = rhoC_ * grad * Partialv;
    grada = rhoC_ * grad * Partiala;
    return true;
  }
  return false;
}

bool TrajOpt::grad_collision_check(const Eigen::Vector3d& p1,
									const Eigen::Vector3d& p2,
                          			Eigen::Vector3d& gradp1,
								    Eigen::Vector3d& gradp2,
                          			double& costp) {
  double dpen = dSwarmMin_ * dSwarmMin_ - (p1-p2).squaredNorm();
  if (dpen > 0) {
    // std::cout<<"dpen: "<<dpen<<std::endl;
    double grad = 0;
    costp = rhoPswarm_ * smoothedTriple(dpen, grad);
    // costp = rhoPswarm_ * penF(dpen, grad);
    gradp1 = - rhoPswarm_ * grad * 2 * (p1 - p2);
	gradp2 = rhoPswarm_ * grad * 2 * (p1 - p2);
    return true;
  }
  return false;
}

//NOTE: there are some bug, to be resolve.
// bool TrajOpt::grad_cost_dirct(const Eigen::Vector3d& v_dir,
//                           Eigen::Vector3d& grad_vdir,
//                           double& cost_vdir) {
//   double pen = 1 - des_theta1_.dot(v_dir)/(v_dir.norm()+1e-6);
//   std::cout<<"pen: "<<pen<<std::endl;
//   if (pen > 0) {
//     double grad = 0;
//     cost_vdir = smoothedL1(pen, grad);
//     grad_vdir -= grad * des_theta1_.transpose() * f_DN(v_dir) ;
//     return true;
//   }
//   return false;
// }

bool TrajOpt::grad_cost_limitv(const double& v,
                          double& gradv,
                          double& costv) {
  double vpen = (v-vmean_) * (v-vmean_) - vgap_ * vgap_;
//   double vpen = v * v - vmax_ * vmax_;
  if (vpen > 0) {
    double grad = 0;
    costv =  rhoVtail_ * smoothedTriple(vpen, grad);
    gradv =  rhoVtail_ * grad * 2 * (v - vmean_);
    return true;
  }
  return false;
}

void TrajOpt::setTargetTraj(Trajectory& target_traj){
    target_traj_poly_ = target_traj;
}

void TrajOpt::setTargetTheta(Eigen::MatrixXd target_theta){
    int num = target_theta.cols();
    des_theta.resize(num);
    for(int i = 0; i < num; i++ )
        des_theta[i] = target_theta.col(i);
}

void TrajOpt::setExcue_T(double& t){
    excue_t = t;
}

TrajOpt::TrajOpt(ros::NodeHandle& nh) {
  nh.getParam("K", K_);
  // load dynamic paramters
  nh.getParam("vmax", vmax_);
  nh.getParam("vmin", vmin_);
  nh.getParam("amax", amax_);
  nh.getParam("amin", amin_);

  nh.getParam("rhoT", rhoT_);
  nh.getParam("rhoV", rhoV_);
  nh.getParam("rhoA", rhoA_);
  nh.getParam("rhoPswarm", rhoPswarm_);
  nh.getParam("rhoVtail", rhoVtail_);
  nh.getParam("rhoC", rhoC_);

  nh.getParam("pause_debug", pause_debug_);
  nh.getParam("monitorUse", monitorUse_);
  nh.getParam("pausems", pausems_);
  nh.getParam("dSwarmMin", dSwarmMin_);
  nh.getParam("omegamax", omegamax_);

  Curmax_ = omegamax_/vmin_;
  visPtr_ = std::make_shared<vis_utils::VisUtils>(nh);

  vmean_ = (vmax_+vmin_)/2;
  vgap_ = (vmax_ - vmin_)/2;
  amean_ = (amax_+amin_)/2;
  agap_ = (amax_ - amin_)/2;  
  v_sqr_mean_ = (vmax_* vmax_ + vmin_ * vmin_)/2;
  v_sqr_gap_ = (vmax_* vmax_ - vmin_ * vmin_)/2;
}

}  // namespace traj_opt