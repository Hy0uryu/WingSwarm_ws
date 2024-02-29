#include <traj_opt/traj_opt.h>

#include <traj_opt/lbfgs_raw.hpp>

namespace traj_opt
{

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

    static bool q2v(const Eigen::Quaterniond &q,
                    Eigen::Vector3d &v)
    {
        Eigen::MatrixXd R = q.toRotationMatrix();
        v = R.col(2);
        return true;
    }
    static Eigen::Vector3d f_N(const Eigen::Vector3d &x)
    {
        return x.normalized();
    }
    static Eigen::MatrixXd f_DN(const Eigen::Vector3d &x)
    {
        double x_norm_2 = x.squaredNorm();
        return (Eigen::MatrixXd::Identity(3, 3) - x * x.transpose() / x_norm_2) / sqrt(x_norm_2);
    }
    static Eigen::MatrixXd f_D2N(const Eigen::Vector3d &x, const Eigen::Vector3d &y)
    {
        double x_norm_2 = x.squaredNorm();
        double x_norm_3 = x_norm_2 * x.norm();
        Eigen::MatrixXd A = (3 * x * x.transpose() / x_norm_2 - Eigen::MatrixXd::Identity(3, 3));
        return (A * y * x.transpose() - x * y.transpose() - x.dot(y) * Eigen::MatrixXd::Identity(3, 3)) / x_norm_3;
    }

    // SECTION  variables transformation and gradient transmission
    static double smoothedL1(const double &x,
                             double &grad)
    {
        static double mu = 0.03;
        if (x < 0.0)
        {
            return 0.0;
        }
        else if (x > mu)
        {
            grad = 1.0;
            return x - 0.5 * mu;
        }
        else
        {
            const double xdmu = x / mu;
            const double sqrxdmu = xdmu * xdmu;
            const double mumxd2 = mu - 0.5 * x;
            grad = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
            return mumxd2 * sqrxdmu * xdmu;
        }
    }

    static double smoothedL1a(const double &x,
                              double &grad)
    {
        static double mu = 0.0003;
        if (x < 0.0)
        {
            return 0.0;
        }
        else if (x > mu)
        {
            grad = 1.0;
            return x - 0.5 * mu;
        }
        else
        {
            const double xdmu = x / mu;
            const double sqrxdmu = xdmu * xdmu;
            const double mumxd2 = mu - 0.5 * x;
            grad = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
            return mumxd2 * sqrxdmu * xdmu;
        }
    }

    static double penF(const double &x, double &grad)
    {
        static double eps = 0.01;
        static double eps2 = eps * eps;
        static double eps3 = eps * eps2;
        if (x < 2 * eps)
        {
            double x2 = x * x;
            double x3 = x * x2;
            double x4 = x2 * x2;
            grad = 12 / eps2 * x2 - 4 / eps3 * x3;
            return 4 / eps2 * x3 - x4 / eps3;
        }
        else
        {
            grad = 16;
            return 16 * (x - eps);
        }
    }

    static double smoothedTriple(const double &x,
                                 double &grad)
    {
        if (x < 0.0)
        {
            return 0.0;
        }
        else
        {
            grad = 3 * x * x;
            return x * x * x;
        }
    }

    static double smoothed01(const double &x,
                             double &grad)
    {
        static double mu = 0.01;
        static double mu4 = mu * mu * mu * mu;
        static double mu4_1 = 1.0 / mu4;
        if (x < -mu)
        {
            grad = 0;
            return 0;
        }
        else if (x < 0)
        {
            double y = x + mu;
            double y2 = y * y;
            grad = y2 * (mu - 2 * x) * mu4_1;
            return 0.5 * y2 * y * (mu - x) * mu4_1;
        }
        else if (x < mu)
        {
            double y = x - mu;
            double y2 = y * y;
            grad = y2 * (mu + 2 * x) * mu4_1;
            return 0.5 * y2 * y * (mu + x) * mu4_1 + 1;
        }
        else
        {
            grad = 0;
            return 1;
        }
    }

    static Eigen::MatrixXd skewMatrix(const Eigen::Vector3d &v)
    {
        Eigen::MatrixXd A(3, 3);
        A << 0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0;
        return A;
    }

    static double expC2(double t)
    {
        return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
                       : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
    }
    static double logC2(double T)
    {
        return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
    }
    static inline double gdT2t(double t)
    {
        if (t > 0)
        {
            return t + 1.0;
        }
        else
        {
            double denSqrt = (0.5 * t - 1.0) * t + 1.0;
            return (1.0 - t) / (denSqrt * denSqrt);
        }
    }

    Eigen::VectorXd TrajOpt::forwardT(const Eigen::VectorXd &t)
    {
        assert(t.size() == N_);
        Eigen::VectorXd T(N_);
        for (int i = 0; i < N_; ++i)
        {
            T(i) = expC2(t(i));
        }
        return T;
    }

    Eigen::VectorXd TrajOpt::backwardT(const Eigen::VectorXd &T)
    {
        assert(T.size() == N_);
        Eigen::VectorXd t(N_);
        for (int i = 0; i < N_; ++i)
        {
            t(i) = logC2(T(i));
        }
        return t;
    }

    Eigen::VectorXd TrajOpt::addLayerTGrad(const Eigen::VectorXd &t,
                                           const Eigen::VectorXd &gradT)
    {
        assert(t.size() == N_);
        Eigen::VectorXd gradt(N_);
        for (int i = 0; i < N_; ++i)
        {
            gradt(i) = gradT(i) * gdT2t(t(i));
        }
        return gradt;
    }

    // the cost function including both smooth part and constraints
    static inline double objectiveFunc_WingSwarm_(void *ptrObj,
                                                  const double *x,
                                                  double *grad,
                                                  const int n)
    {
        iter_times_++;
        TrajOpt &obj = *(TrajOpt *)ptrObj;
        const double &x_t = x[0];

        Eigen::Map<const Eigen::MatrixXd> x_p(x + obj.dim_t_, 3, obj.drone_num_ * obj.dim_p_);
        Eigen::Map<const Eigen::VectorXd> v_tail(x + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_, obj.dim_tail_);
        Eigen::Map<const Eigen::VectorXd> a_tail(x + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_ + obj.dim_tail_, obj.dim_atail_);
        Eigen::Map<const Eigen::VectorXd> j_tail(x + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_ + obj.dim_tail_ + obj.dim_atail_, obj.dim_jtail_);

        double &gradt = grad[0];
        Eigen::Map<Eigen::MatrixXd> gradp(grad + obj.dim_t_, 3, obj.drone_num_ * obj.dim_p_);
        Eigen::Map<Eigen::VectorXd> gradvtail(grad + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_, obj.dim_tail_);
        Eigen::Map<Eigen::VectorXd> gradatail(grad + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_ + obj.dim_tail_, obj.dim_atail_);
        Eigen::Map<Eigen::VectorXd> gradjtail(grad + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_ + obj.dim_tail_ + obj.dim_atail_, obj.dim_jtail_);

        double p_t = obj.TimeNoSafe_ / double(obj.N_);
        double dT = expC2(x_t) + p_t;
        // std::cout<<"x_t: "<<x_t<<std::endl;
        // std::cout<<"expC2(x_t): "<<expC2(x_t)<<std::endl;
        // std::cout<<"outside dT: "<<dT<<std::endl;
        double end_time = dT * obj.N_ + obj.excue_t;
        obj.qpos = obj.target_traj_poly_.getPos(end_time);
        obj.qvel = obj.target_traj_poly_.getVel(end_time);

        double cost = 0;
        obj.swarm_gdT = 0;
        // get the trajectory generated from MINCO and caculate the smooth cost
        for (int i = 0; i < obj.drone_num_; i++)
        {
            // set the tail state
            obj.initE_[i].col(0) = obj.target_traj_poly_.getPos(end_time);
            obj.initE_[i].col(1) = v_tail[i] * obj.des_theta[i];
            obj.initE_[i].col(2) = a_tail.middleRows(i * 3, 3);
            obj.initE_[i].col(3) = j_tail.middleRows(i * 3, 3);
            // obj.initE_[i].col(2).setZero();
            // obj.initE_[i].col(3).setZero();

            obj.swarm_mincoOpt_[i].generate(obj.initS_[i], obj.initE_[i], x_p.block(0, i * obj.dim_p_, 3, obj.dim_p_), dT);
            cost += obj.swarm_mincoOpt_[i].getTrajSnapCost();
            obj.swarm_mincoOpt_[i].calGrads_CT();
        }

        // get the costs and gradients related to time integral
        obj.addTimeIntPenalty_Swarm(cost);
        obj.addTimeIntPenalty_Swarm_inFixedTimeStep(cost);

        gradt = 0;
        gradvtail.setZero();
        gradatail.setZero();
        gradjtail.setZero();

        // get the costs and gradients related to terminal state
        for (int i = 0; i < obj.drone_num_; i++)
        {
            obj.swarm_mincoOpt_[i].calGrads_PT();
            gradp.block(0, i * obj.dim_p_, 3, obj.dim_p_) = obj.swarm_mincoOpt_[i].gdP;

            double PatialrhoToTime = 0;
            PatialrhoToTime += obj.swarm_mincoOpt_[i].gdTail.col(0).dot(obj.N_ * obj.target_traj_poly_.getVel(end_time));
            obj.swarm_mincoOpt_[i].gdT += PatialrhoToTime;

            gradvtail(i) += obj.swarm_mincoOpt_[i].gdTail.col(1).dot(obj.des_theta[i]);
            gradatail.middleRows(3 * i, 3) += obj.swarm_mincoOpt_[i].gdTail.col(2);
            gradjtail.middleRows(3 * i, 3) = obj.swarm_mincoOpt_[i].gdTail.col(3);

            double grad_tmp, cost_vlimit;
            if (obj.grad_cost_limitv(v_tail(i), grad_tmp, cost_vlimit))
            {
                gradvtail(i) += grad_tmp;
                cost += cost_vlimit;
            }

            double cost_alimit;
            Eigen::Vector3d grad_atmp;
            if (obj.grad_cost_limita(obj.des_theta[i], a_tail.middleRows(3 * i, 3), grad_atmp, cost_alimit))
            {
                gradatail.middleRows(3 * i, 3) += grad_atmp;
                cost += cost_alimit;
            }

            gradt += obj.swarm_mincoOpt_[i].gdT * gdT2t(x_t);
        }

        // sum the all cost and gradients
        gradt += obj.swarm_gdT * gdT2t(x_t);

        gradt += obj.rhoT_ * gdT2t(x_t) * obj.N_;

        // gradt = 0;
        // gradp.setZero();
        cost += obj.rhoT_ * dT * obj.N_;

        return cost;
    }

    static inline int processMonitor(void *ptrObj,
                                     const double *x,
                                     const double *grad,
                                     const double fx,
                                     const double xnorm,
                                     const double gnorm,
                                     const double step,
                                     int n,
                                     int k,
                                     int ls)
    {
        TrajOpt &obj = *(TrajOpt *)ptrObj;
        if (obj.monitorUse_)
        {
            const double x_t = x[0];
            Eigen::Map<const Eigen::MatrixXd> x_p(x + obj.dim_t_, 3, obj.drone_num_ * obj.dim_p_);
            Eigen::Map<const Eigen::VectorXd> v_tail(x + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_, obj.dim_tail_);
            Eigen::Map<const Eigen::VectorXd> a_tail(x + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_ + obj.dim_tail_, obj.dim_atail_);
            Eigen::Map<const Eigen::VectorXd> j_tail(x + obj.dim_t_ + 3 * obj.drone_num_ * obj.dim_p_ + obj.dim_tail_ + obj.dim_atail_, obj.dim_jtail_);

            double dT = expC2(x_t);

            double end_time = dT * obj.N_ + obj.excue_t;
            std::string id_sample;

            for (int i = 0; i < obj.drone_num_; i++)
            {
                obj.visPtr_->visualize_traj(obj.target_traj_poly_, "target_traj_zyh");

                obj.initE_Debug_[i].col(0) = obj.target_traj_poly_.getPos(end_time);
                obj.initE_Debug_[i].col(1) = v_tail[i] * obj.des_theta[i];
                obj.initE_Debug_[i].col(2) = a_tail.middleRows(i * 3, 3);
                obj.initE_Debug_[i].col(3) = j_tail.middleRows(i * 3, 3);

                obj.swarm_debugOpt_[i].generate(obj.initS_[i], obj.initE_Debug_[i], x_p.block(0, i * obj.dim_p_, 3, obj.dim_p_), dT);
                id_sample = "optimizing_traj_" + std::to_string(i);
                obj.visPtr_->visualize_traj(obj.swarm_debugOpt_[i].getTraj(), id_sample);
            }

            // NOTE pause
            std::this_thread::sleep_for(std::chrono::milliseconds(obj.pausems_));
        }
        return 0;
    }

    bool TrajOpt::generate_traj(const Eigen::MatrixXd &initState,
                                const Eigen::MatrixXd &innerPts,
                                std::vector<Trajectory> &swarm_traj_,
                                const double &init_t)
    {
        assert(initState.cols() % 4 == 0);
        drone_num_ = initState.cols() / 4;
        N_ = innerPts.cols() / drone_num_ + 1;
        dim_t_ = 1;
        dim_p_ = N_ - 1;
        dim_tail_ = drone_num_;
        dim_atail_ = drone_num_ * 3;
        dim_jtail_ = drone_num_ * 3;
        assert(N_ > 1);
        std::cout << "innerPts_num_:" << innerPts.cols() << std::endl;

        x_ = new double[dim_t_ + 3 * drone_num_ * dim_p_ + dim_tail_ + dim_atail_ + dim_jtail_];
        double &x_t = x_[0];

        Eigen::Map<Eigen::MatrixXd> innerp_set(x_ + dim_t_, 3, drone_num_ * dim_p_);
        Eigen::Map<Eigen::VectorXd> v_tail(x_ + dim_t_ + 3 * drone_num_ * dim_p_, dim_tail_);
        Eigen::Map<Eigen::VectorXd> a_tail(x_ + dim_t_ + 3 * drone_num_ * dim_p_ + dim_tail_, dim_atail_);
        Eigen::Map<Eigen::VectorXd> j_tail(x_ + dim_t_ + 3 * drone_num_ * dim_p_ + dim_tail_ + dim_atail_, dim_jtail_);

        innerp_set = innerPts;

        swarm_mincoOpt_.resize(drone_num_);
        swarm_debugOpt_.resize(drone_num_);
        initS_.resize(drone_num_);
        initE_.resize(drone_num_);
        initE_Debug_.resize(drone_num_);

        for (int i = 0; i < drone_num_; i++)
        {
            swarm_mincoOpt_[i].reset(N_);
            swarm_debugOpt_[i].reset(N_);
            initS_[i].resize(3, 4);
            initE_[i].resize(3, 4);
            initE_Debug_[i].resize(3, 4);
            initS_[i] = initState.block<3, 4>(0, i * 4);
        }

        double p_t = TimeNoSafe_ / double(N_);
        if(init_t - p_t<0)x_t = logC2(0.1);
        else x_t = logC2(init_t - p_t);
        v_tail.fill(vmax_);
        a_tail.fill(0);
        j_tail.fill(0);

        /* Init Data Print
        std::cout<<"optimization start!"<<std::endl;
        std::cout<<"total num: "<<drone_num_<<std::endl;
        for(int i =0; i< drone_num_; i++){
            std::cout<<"drone_num: ["<<i<<"]"<<std::endl;
            std::cout<<"INIT STATE:"<<std::endl;
            std::cout<<initS_[i].transpose()<<std::endl;
            std::cout<<"INTERIOR POINTS:"<<std::endl;
            std::cout<<innerp_set.block(0, i * dim_p_ , 3, dim_p_).transpose()<<std::endl;
        }
        std::cout<<"init_T: "<<ds/vmax_<<std::endl;
        */

        // NOTE optimization LBFGS param set
        lbfgs::lbfgs_parameter_t lbfgs_params;
        lbfgs::lbfgs_load_default_parameters(&lbfgs_params);
        // lbfgs_params.mem_size = 128; // 64
        // lbfgs_params.past = 3;
        // lbfgs_params.delta = 1e-6;
        // lbfgs_params.g_epsilon = 0.01;
        // lbfgs_params.min_step = 1e-64;
        // lbfgs_params.line_search_type = 0; // 0

        lbfgs_params.mem_size = 256; // 64
        lbfgs_params.past = 0;
        lbfgs_params.delta = 1e-6;
        lbfgs_params.g_epsilon = 0.1;
        lbfgs_params.min_step = 1e-128;
        lbfgs_params.line_search_type = 0; // 0

        double minObjective;

        int opt_ret = 0;

        std::cout << "******************************" << std::endl;

        auto tic = std::chrono::steady_clock::now();

        // Optimize the trajectory by LBFGS
        opt_ret = lbfgs::lbfgs_optimize(dim_t_ + 3 * drone_num_ * dim_p_ + dim_tail_ + dim_atail_ + dim_jtail_, x_, &minObjective,
                                        &objectiveFunc_WingSwarm_, nullptr,
                                        &processMonitor, this, &lbfgs_params);

        auto toc = std::chrono::steady_clock::now();

        std::cout << "\033[32m>ret: " << opt_ret << "\033[0m" << std::endl;

        std::cout << "optmization costs: " << (toc - tic).count() * 1e-6 << "ms" << std::endl;

        std::cout << "iterative times: " << iter_times_ << std::endl;

        if (opt_ret < 0)
        {
            delete[] x_;
            return false;
        }

        for (int i = 0; i < swarm_traj_.size(); i++)
            swarm_traj_[i] = swarm_mincoOpt_[i].getTraj();

        delete[] x_;
        return true;
    }

    void TrajOpt::addTimeIntPenalty_Swarm(double &cost)
    {
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
        pos.resize(3, drone_num_);
        vel.resize(3, drone_num_);
        acc.resize(3, drone_num_);
        jer.resize(3, drone_num_);
        snp.resize(3, drone_num_);
        grad_p.resize(3, drone_num_);
        grad_v.resize(3, drone_num_);
        grad_a.resize(3, drone_num_);
        grad_j.resize(3, drone_num_);
        cost_inner.resize(drone_num_);
        gradViola_c.resize(drone_num_);

        int innerLoop = K_ + 1;
        step = swarm_mincoOpt_[0].t(1) / K_;

        s1 = 0.0;
        for (int j = 0; j < innerLoop; ++j)
        {
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

            for (int i = 0; i < N_; ++i)
            {
                std::vector<Eigen::Matrix<double, 8, 3>> c(drone_num_);
                for (int k = 0; k < drone_num_; k++)
                {
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
                for (int k = 0; k < drone_num_; k++)
                {
                    if (true)
                    {
                        if (grad_cost_v(vel.col(k), grad_tmp, cost_tmp))
                        {
                            grad_v.col(k) += grad_tmp;
                            cost_inner(k) += cost_tmp;
                        }

                        Eigen::Vector3d grad_tmp1, grad_tmp2;
                        if (grad_cost_a(acc.col(k), vel.col(k), grad_tmp1, grad_tmp2, cost_tmp))
                        {
                            grad_v.col(k) += grad_tmp2;
                            grad_a.col(k) += grad_tmp1;
                            cost_inner(k) += cost_tmp;
                        }

                        if (grad_curvature_check(acc.col(k), vel.col(k), grad_tmp1, grad_tmp2, cost_tmp))
                        {
                            // std::cout<<cost_tmp<<" "<<grad_tmp1.transpose()<<" "<<grad_tmp2.transpose()<<std::endl;
                            grad_v.col(k) += grad_tmp2;
                            grad_a.col(k) += grad_tmp1;
                            cost_inner(k) += cost_tmp;
                        }
                    }
                }

                // penalty among the drones
                // if (i < N_-1)
                // {
                //     Eigen::Vector3d grad_tmp1, grad_tmp2;
                //     double cost_swarm;
                //     Eigen::Vector3d gradq;
                //     for (int x = 0; x < drone_num_; x++)
                //         for (int y = x + 1; y < drone_num_; y++)
                //         {
                //             if(grad_collision_check(pos.col(x),pos.col(y),grad_tmp1,grad_tmp2,cost_swarm)){
                //             	grad_p.col(x) += grad_tmp1;
                //             	grad_p.col(y) += grad_tmp2;
                //             	swarm_gdT += omg * cost_swarm / K_;
                //             	cost += omg * step * cost_swarm;
                //             }
                //             // if (grad_collision_check(pos.col(x), pos.col(y), qpos, grad_tmp1, grad_tmp2, gradq, cost_swarm))
                //             // {
                //             //     grad_p.col(x) += grad_tmp1;
                //             //     grad_p.col(y) += grad_tmp2;
                //             //     swarm_gdT += omg * cost_swarm / K_ + omg * step * N_ * gradq.dot(qvel);
                //             //     cost += omg * step * cost_swarm;
                //             // }
                //         }
                // }

                for (int k = 0; k < drone_num_; k++)
                {
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

    void TrajOpt::addTimeIntPenalty_Swarm_inFixedTimeStep(double &cost)
    {
        Eigen::MatrixXd pos, vel, acc, jer, snp;
        Eigen::MatrixXd grad_p, grad_v, grad_a, grad_j;
        Eigen::Vector3d grad_tmp;
        double cost_tmp;
        Eigen::VectorXd cost_inner;
        Eigen::Matrix<double, 8, 1> beta0, beta1, beta2, beta3, beta4;
        double s1, s2, s3, s4, s5, s6, s7;
        double step, alpha;
        double NK_ = 0, jNK_ = 0;
        std::vector<Eigen::Matrix<double, 8, 3>> gradViola_c;
        double gradViola_t;
        double omg;
        pos.resize(3, drone_num_);
        vel.resize(3, drone_num_);
        acc.resize(3, drone_num_);
        jer.resize(3, drone_num_);
        snp.resize(3, drone_num_);
        grad_p.resize(3, drone_num_);
        grad_v.resize(3, drone_num_);
        grad_a.resize(3, drone_num_);
        grad_j.resize(3, drone_num_);
        cost_inner.resize(drone_num_);
        gradViola_c.resize(drone_num_);
        double dT = swarm_mincoOpt_[0].t(1);

        /*T-Tf*/
        int K_ = FixK_;
        double signalTime = TimeNoSafe_;
        double intTime = dT * N_ - signalTime;
        step = intTime / K_;
        NK_ = double(N_) / double(K_);

        int IdxPiece = 0;
        double t = 0.0;
        double t_pre = 0;

        for (int j = 0; j < K_ + 1; ++j)
        {
            while(t - t_pre > dT){
                t_pre += dT;
                IdxPiece ++;
            }

            s1 = t -t_pre;
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

            omg = (j == 0 || j == K_) ? 0.5 : 1.0;
            jNK_ = j * NK_;

            std::vector<Eigen::Matrix<double, 8, 3>> c(drone_num_);
            for (int k = 0; k < drone_num_; k++)
            {
                c[k] = swarm_mincoOpt_[k].c.block<8, 3>(IdxPiece * 8, 0);
                pos.col(k) = c[k].transpose() * beta0;
                vel.col(k) = c[k].transpose() * beta1;

                acc.col(k) = c[k].transpose() * beta2;
                jer.col(k) = c[k].transpose() * beta3;
                snp.col(k) = c[k].transpose() * beta4;
            }

            grad_p.setZero();

            // penalty among the drones
            Eigen::Vector3d grad_tmp1, grad_tmp2;
            double cost_swarm;
            Eigen::Vector3d gradq;
            for(int x = 0; x < drone_num_; x++)
            		for( int y = x+1; y < drone_num_; y++){
            				if(grad_collision_check(pos.col(x),pos.col(y),grad_tmp1,grad_tmp2,cost_swarm)){
            					grad_p.col(x) += grad_tmp1;
            					grad_p.col(y) += grad_tmp2;
            					cost += omg * step * cost_swarm;
                                swarm_gdT += omg * NK_ * cost_swarm;
                            }
            }

            for (int k = 0; k < drone_num_; k++)
            {
                gradViola_c[k] = beta0 * grad_p.col(k).transpose();
                gradViola_t = grad_p.col(k).transpose() * vel.col(k);

                {
                swarm_mincoOpt_[k].gdC.block<8, 3>(IdxPiece * 8, 0) += omg * step * gradViola_c[k];

                /*T-Tf*/
                if(IdxPiece > 0){
                    swarm_mincoOpt_[k].gdT += omg * step * gradViola_t * (jNK_ - IdxPiece);
                } else
                    swarm_mincoOpt_[k].gdT += omg * step * gradViola_t * (jNK_);
                }
            }
            t += step;
        }

        /*T-Tf only
        {
            t = dT * N_ - signalTime;
            while(t - t_pre > dT){
                t_pre += dT;
                IdxPiece ++;
            }           
                    
            s1 = t - t_pre;
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

            std::vector<Eigen::Matrix<double, 8, 3>> c(drone_num_);
            for (int k = 0; k < drone_num_; k++)
            {
                c[k] = swarm_mincoOpt_[k].c.block<8, 3>(IdxPiece * 8, 0);
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
            for (int k = 0; k < drone_num_; k++)
            {
                if (true)
                {
                    if (grad_cost_v(vel.col(k), grad_tmp, cost_tmp))
                    {
                        grad_v.col(k) += grad_tmp;
                        cost_inner(k) += cost_tmp;
                    }

                    Eigen::Vector3d grad_tmp1, grad_tmp2;
                    if (grad_cost_a(acc.col(k), vel.col(k), grad_tmp1, grad_tmp2, cost_tmp))
                    {
                        grad_v.col(k) += grad_tmp2;
                        grad_a.col(k) += grad_tmp1;
                        cost_inner(k) += cost_tmp;
                    }

                    if (grad_curvature_check(acc.col(k), vel.col(k), grad_tmp1, grad_tmp2, cost_tmp))
                    {
                        grad_v.col(k) += grad_tmp2;
                        grad_a.col(k) += grad_tmp1;
                        cost_inner(k) += cost_tmp;
                    }
                }
            }

            for (int k = 0; k < drone_num_; k++)
            {
                gradViola_c[k] = beta0 * grad_p.col(k).transpose();
                gradViola_t = grad_p.col(k).transpose() * vel.col(k);
                gradViola_c[k] += beta1 * grad_v.col(k).transpose();
                gradViola_t += grad_v.col(k).transpose() * acc.col(k);
                gradViola_c[k] += beta2 * grad_a.col(k).transpose();
                gradViola_t += grad_a.col(k).transpose() * jer.col(k);
                gradViola_c[k] += beta3 * grad_j.col(k).transpose();
                gradViola_t += grad_j.col(k).transpose() * snp.col(k);
                swarm_mincoOpt_[k].gdC.block<8, 3>(IdxPiece * 8, 0) += gradViola_c[k];

                //T-Tf
                if(IdxPiece > 0){
                    swarm_mincoOpt_[k].gdT += gradViola_t * (N_ - IdxPiece);
                    std::cout<<"11111"<<std::endl;
                } else
                    swarm_mincoOpt_[k].gdT += gradViola_t * (N_);

                cost += cost_inner(k);
            }
        }
        */

    }

    bool TrajOpt::grad_cost_v(const Eigen::Vector3d &v,
                              Eigen::Vector3d &gradv,
                              double &costv)
    {
        double vpen = (v.squaredNorm() - v_sqr_mean_) * (v.squaredNorm() - v_sqr_mean_) - v_sqr_gap_ * v_sqr_gap_;
        if (vpen > 0)
        {
            double grad = 0;
            // costv = rhoV_ * smoothedTriple(vpen, grad);
            costv = rhoV_ * smoothedL1(vpen, grad);
            gradv = rhoV_ * grad * 4 * (v.squaredNorm() - v_sqr_mean_) * v;
            return true;
        }
        return false;
    }

    bool TrajOpt::grad_cost_a(const Eigen::Vector3d &a,
                              const Eigen::Vector3d &v,
                              Eigen::Vector3d &grada,
                              Eigen::Vector3d &gradv,
                              double &costa)
    {
        double apen;
        apen = v.normalized().dot(a) - amax_;
        double grad = 0;
        if (apen > 0)
        {
            // costa = rhoA_ * smoothedTriple(apen, grad);
            costa = rhoA_ * smoothedL1a(apen, grad);
            gradv = rhoA_ * grad * f_DN(v) * a;
            grada = rhoA_ * grad * v.normalized();
            return true;
        }
        else
        {
            apen = amin_ - v.normalized().dot(a);
            if (apen > 0)
            {
                // costa = rhoA_ * smoothedTriple(apen, grad);
                costa = rhoA_ * smoothedL1a(apen, grad);
                gradv = -rhoA_ * grad * f_DN(v) * a;
                grada = -rhoA_ * grad * v.normalized();
                return true;
            }
        }
        return false;
    }

    bool TrajOpt::grad_cost_limita(const Eigen::Vector3d &theta,
                                   const Eigen::Vector3d &a,
                                   Eigen::Vector3d &grada,
                                   double &costa)
    {
        double apen = theta.dot(a) - amax_;
        double grad = 0;
        if (apen > 0)
        {
            costa = rhoAtail_ * smoothedL1a(apen, grad);
            grada = rhoAtail_ * grad * theta;
            return true;
        }
        else
        {
            apen = amin_ - theta.dot(a);
            if (apen > 0)
            {
                costa = rhoAtail_ * smoothedL1a(apen, grad);
                grada = -rhoAtail_ * grad * theta;
                return true;
            }
        }
        return false;
    }

    bool TrajOpt::grad_curvature_check(const Eigen::Vector3d &a,
                                       const Eigen::Vector3d &v,
                                       Eigen::Vector3d &grada,
                                       Eigen::Vector3d &gradv,
                                       double &cost)
    {

        double pen;
        pen = (v.cross(a)).norm() / pow(v.norm(), 3) - Curmax_;
        if (pen > 0)
        {
            double grad = 0;
            // cost = rhoC_ * smoothedTriple(pen, grad);
            cost = rhoC_ * smoothedL1(pen, grad);
            Eigen::Vector3d Partialv = -(1 / ((v.cross(a)).norm() * pow(v.norm(), 3))) * skewMatrix(a).transpose() * (v.cross(a)) - 3 * (v.cross(a)).norm() / pow(v.norm(), 5) * v;
            Eigen::Vector3d Partiala = (1 / (pow(v.norm(), 3) * (v.cross(a)).norm())) * skewMatrix(v).transpose() * (v.cross(a));
            gradv = rhoC_ * grad * Partialv;
            grada = rhoC_ * grad * Partiala;
            return true;
        }
        return false;
    }

    bool TrajOpt::grad_collision_check(const Eigen::Vector3d &p1,
                                       const Eigen::Vector3d &p2,
                                       const Eigen::Vector3d &endp,
                                       Eigen::Vector3d &gradp1,
                                       Eigen::Vector3d &gradp2,
                                       Eigen::Vector3d &gradq,
                                       double &costp)
    {
        double dpen = dSwarmMin_ * dSwarmMin_ - (p1 - p2).squaredNorm();
        double d_threshold = 4;
        double despen1 = (p1 - endp).squaredNorm() - d_threshold;
        double despen2 = (p2 - endp).squaredNorm() - d_threshold;
        if (despen1 < despen2)
        {
            if (dpen > 0)
            {
                double grad = 0;
                double grad01 = 0;
                double cost1 = smoothed01(despen1, grad01);
                double cost2 = smoothedL1(dpen, grad);
                costp = rhoPswarm_ * cost1 * cost2;
                gradp1 = rhoPswarm_ * (-cost1 * grad * 2 * (p1 - p2) + grad01 * 2 * (p1 - endp) * cost2);
                gradp2 = rhoPswarm_ * cost1 * grad * 2 * (p1 - p2);
                gradq = -rhoPswarm_ * cost2 * grad01 * 2 * (p1 - endp);
                return true;
            }
        }
        else
        {
            if (dpen > 0)
            {
                double grad = 0;
                double grad01 = 0;
                double cost1 = smoothed01(despen2, grad01);
                double cost2 = smoothedL1(dpen, grad);
                costp = rhoPswarm_ * cost1 * cost2;
                gradp1 = -rhoPswarm_ * cost1 * grad * 2 * (p1 - p2);
                gradp2 = rhoPswarm_ * (cost1 * grad * 2 * (p1 - p2) + +grad01 * 2 * (p2 - endp) * cost2);
                gradq = -rhoPswarm_ * cost2 * grad01 * 2 * (p2 - endp);
                return true;
            }
        }
        return false;
    }

    bool TrajOpt::grad_collision_check(const Eigen::Vector3d &p1,
                                       const Eigen::Vector3d &p2,
                                       Eigen::Vector3d &gradp1,
                                       Eigen::Vector3d &gradp2,
                                       double &costp)
    {
        double dpen = dSwarmMin_ * dSwarmMin_ - (p1 - p2).squaredNorm();
        if (dpen > 0)
        {
            // std::cout<<"dpen: "<<dpen<<std::endl;
            double grad = 0;
            costp = rhoPswarm_ * smoothedL1(dpen, grad);
            // costp = rhoPswarm_ * penF(dpen, grad);
            gradp1 = -rhoPswarm_ * grad * 2 * (p1 - p2);
            gradp2 = rhoPswarm_ * grad * 2 * (p1 - p2);
            return true;
        }
        return false;
    }

    bool TrajOpt::grad_cost_limitv(const double &v,
                                   double &gradv,
                                   double &costv)
    {
        double vpen = (v - vmean_) * (v - vmean_) - vgap_ * vgap_;
        //   double vpen = v * v - vmax_ * vmax_;
        if (vpen > 0)
        {
            double grad = 0;
            costv = rhoVtail_ * smoothedTriple(vpen, grad);
            gradv = rhoVtail_ * grad * 2 * (v - vmean_);
            return true;
        }
        return false;
    }

    void TrajOpt::setTargetTraj(Trajectory &target_traj)
    {
        target_traj_poly_ = target_traj;
    }

    void TrajOpt::setTargetTheta(Eigen::MatrixXd target_theta)
    {
        int num = target_theta.cols();
        des_theta.resize(num);
        for (int i = 0; i < num; i++)
            des_theta[i] = target_theta.col(i);
    }

    void TrajOpt::setExcue_T(double &t)
    {
        excue_t = t;
    }

    TrajOpt::TrajOpt(ros::NodeHandle &nh)
    {
        nh.getParam("K", K_);
        // load dynamic paramters
        nh.getParam("vmax", vmax_);
        nh.getParam("vmin", vmin_);
        nh.getParam("amax", amax_);
        nh.getParam("amin", amin_);

        nh.getParam("rhoT", rhoT_);
        nh.getParam("rhoV", rhoV_);
        nh.getParam("rhoA", rhoA_);
        nh.getParam("rhoAtail", rhoAtail_);
        nh.getParam("rhoPswarm", rhoPswarm_);
        nh.getParam("rhoVtail", rhoVtail_);
        nh.getParam("rhoC", rhoC_);

        nh.getParam("monitorUse", monitorUse_);
        nh.getParam("pausems", pausems_);
        nh.getParam("dSwarmMin", dSwarmMin_);
        nh.getParam("omegamax", omegamax_);

        nh.getParam("TimeNoSafe",TimeNoSafe_);
        nh.getParam("FixK",FixK_);

        Curmax_ = omegamax_ / vmin_;
        visPtr_ = std::make_shared<vis_utils::VisUtils>(nh);

        vmean_ = (vmax_ + vmin_) / 2;
        vgap_ = (vmax_ - vmin_) / 2;
        amean_ = (amax_ + amin_) / 2;
        agap_ = (amax_ - amin_) / 2;
        v_sqr_mean_ = (vmax_ * vmax_ + vmin_ * vmin_) / 2;
        v_sqr_gap_ = (vmax_ * vmax_ - vmin_ * vmin_) / 2;
    }

} // namespace traj_opt