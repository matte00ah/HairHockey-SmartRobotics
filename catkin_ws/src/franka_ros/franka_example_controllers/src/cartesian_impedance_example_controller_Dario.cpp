 // Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/cartesian_impedance_example_controller_Dario.h>


#include <cmath>
#include <memory>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include <franka_example_controllers/pseudo_inversion.h>


#include <std_msgs/Float64MultiArray.h>

#include <std_msgs/Float64.h>
#include <std_msgs/Float64.h>
#include <franka/model.h>
#include <franka/robot.h>

#include <iostream>
#include <geometry_msgs/AccelStamped.h>

#include <std_msgs/Bool.h>

ros::Publisher pose_publisher;
// Aggiungi questo vicino alle dichiarazioni degli altri publisher
ros::Publisher F_estimated_pub;


namespace {
template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}  // anonymous namespace  
  

namespace franka_example_controllers {

//----------------------------------------------------------------------------- 
bool CartesianImpedanceExampleController_Dario::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle)
//----------------------------------------------------------------------------- 
{
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;
  std::vector<double> cartesian_inertia_vector;
  std::vector<double> cartesian_inertia_vector_inv;


  sub_equilibrium_pose_ = node_handle.subscribe(
      "equilibrium_pose", 20, &CartesianImpedanceExampleController_Dario::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());
       
  sub_force_sensor_ = node_handle.subscribe(
      "/netft_data", 20, &CartesianImpedanceExampleController_Dario::forcesensorCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());
  
  // Assicurati che il nome coincida con quello che useremo nello script python
  sub_reset_bias_ = node_handle.subscribe(
    "reset_bias", 1, &CartesianImpedanceExampleController_Dario::resetBiasCallback, this);

  K_pub = node_handle.advertise<std_msgs::Float64>("/kinetic_energy", 10);
  P_pub = node_handle.advertise<geometry_msgs::Point>("/x_tj", 10, this );
  V_pub = node_handle.advertise<geometry_msgs::TwistStamped>("/v_tj", 10, this );
  A_pub = node_handle.advertise<geometry_msgs::AccelStamped>("/a_tj", 10, this );
  F_pub = node_handle.advertise<geometry_msgs::TwistStamped>("/F_EE", 10, this );
  m_pub = node_handle.advertise<geometry_msgs::Point>("/m", 10, this );
  // Inizializza il nuovo publisher
  F_estimated_pub = node_handle.advertise<geometry_msgs::TwistStamped>("/F_EE_estimated", 10, this);
F_base_pub = node_handle.advertise<geometry_msgs::TwistStamped>("/F_ext_base", 10, this);


  sub_desired_velocity_ = node_handle.subscribe(
      "desired_velocity", 20, &CartesianImpedanceExampleController_Dario::DesiredVelocitySubscriberCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());
  
  sub_desired_acceleration_ = node_handle.subscribe(
      "desired_acceleration", 20, &CartesianImpedanceExampleController_Dario::DesiredAccelerationSubscriberCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());
  
  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("CartesianImpedanceExampleController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "CartesianImpedanceExampleController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  auto* model_interface = robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_ = std::make_unique<franka_hw::FrankaModelHandle>(
        model_interface->getHandle(arm_id + "_model"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  auto* state_interface = robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_ = std::make_unique<franka_hw::FrankaStateHandle>(
        state_interface->getHandle(arm_id + "_robot"));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  auto* effort_joint_interface = robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "CartesianImpedanceExampleController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "CartesianImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle(node_handle.getNamespace() + "/dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_ = std::make_unique<
      dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>>(
      dynamic_reconfigure_compliance_param_node_);
      
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&CartesianImpedanceExampleController_Dario::complianceParamCallback, this, _1, _2));

  double init_translational_stiffness = 500.0;
  double init_rotational_stiffness = 50.0;

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  cartesian_stiffness_.setZero();
  cartesian_stiffness_.topLeftCorner(3, 3) << init_translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_.bottomRightCorner(3, 3) << init_rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_ = cartesian_stiffness_; // Il target parte già allineato
  
  double init_translational_damping = 0.5 * sqrt(2.0 * init_translational_stiffness);
  double init_rotational_damping = 0.5 * sqrt(2.0 * init_rotational_stiffness);
  
  cartesian_damping_.setZero();
  cartesian_damping_.topLeftCorner(3, 3) << init_translational_damping * Eigen::Matrix3d::Identity();
  cartesian_damping_.bottomRightCorner(3, 3) << init_rotational_damping * Eigen::Matrix3d::Identity();
  cartesian_damping_target_ = cartesian_damping_;

  cartesian_inertia_.setIdentity();      
  cartesian_inertia_target_.setIdentity();
  cartesian_inertia_inv_.setIdentity();       // [FIX] Meglio Identity che Zero per l'inversa
  cartesian_inertia_target_inv_.setIdentity();
  //pose_publisher = node_handle.advertise<std_msgs::Float64MultiArray>("pose_topic", 10);
  
  nullspace_stiffness_ = 20.0; // Valore standard sicuro
  nullspace_stiffness_target_ = 20.0;
  
  // --- AGGIUNTA PER SOFT BIAS ---
  wrench_bias_.setZero();
  is_bias_initialized_ = false; // Forza il ricalcolo del bias all'avvio
  // ------------------------------
  
  // Aggiungi in fondo a init:
  velocity_prev_.setZero();
  // Dentro init(), vicino agli altri setZero:
  dq_prev_.setZero();
  acc_filtered_.setZero();
  velocity_desired.setZero();
  acceleration_desired.setZero();
  
  // [FIX] Inizializza J_dot (era un'altra causa di crash)
  J_dot.setZero();

  return true;
}
//END init

//----------------------------------------------------------------------------- 
void CartesianImpedanceExampleController_Dario::starting(const ros::Time& /*time*/)
//----------------------------------------------------------------------------- 
{
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  position_d_            = initial_transform.translation();
  orientation_d_         = Eigen::Quaterniond(initial_transform.rotation());
  position_d_target_     = initial_transform.translation();
  orientation_d_target_  = Eigen::Quaterniond(initial_transform.rotation());
  orientation_d_target_0 = Eigen::Quaterniond(initial_transform.rotation());
  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ = q_initial;
  
  std::array<double, 16> x = model_handle_->getPose(franka::Frame::kEndEffector);

  p_prev[0]=x[12];
  p_prev[1]=x[13];
  p_prev[2]=x[14];
  
  velocity_prev_.setZero();
  
  // Aggiungi in fondo a starting:
  velocity_desired.setZero();
  acceleration_desired.setZero();
  
  // --- AGGIUNTA FONDAMENTALE ---
  J_dot.setZero(); // <--- AGGIUNGI QUESTO
  Eigen::Map<Eigen::Matrix<double, 6, 7>> current_jacobian(jacobian_array.data()); 
  prev_jacobian = current_jacobian; 
  // -----------------------------
  
  Eigen::Map<Eigen::Matrix<double, 7, 1>> initial_dq(initial_state.dq.data());
  dq_prev_ = initial_dq;
  acc_filtered_.setZero();
  
//END starting
}



//----------------------------------------------------------------------------- 
void CartesianImpedanceExampleController_Dario::update(const ros::Time& /*time*/, const ros::Duration& period)
//----------------------------------------------------------------------------- 
{  //prima era così /*period*/ 
  
  franka::RobotState robot_state        = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array  = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  std::array<double, 49> mass_array = model_handle_->getMass();
	
  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1>> tau_J_d(robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());

  Eigen::Map<Eigen::Matrix<double, 7, 7>> M_q(mass_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 1>> Fee(robot_state.O_F_ext_hat_K.data());
  

  //-----------------------
  // PUBLISHERS  
  //----------------------- 

  // Estrai la matrice di rotazione corrente (EE rispetto alla Base)
  R_EE_to_Base_ = transform.rotation();

  // Applica la rotazione alle Forze (head) e alle Coppie (tail)
  // wrench_measured viene dal sensore (frame EE), wrench_base_frame_ sarà nel frame Base
  wrench_base_frame_.head(3) = R_EE_to_Base_ * wrench_measured.head(3);
  wrench_base_frame_.tail(3) = R_EE_to_Base_ * wrench_measured.tail(3);

  // ----------------------------------------------------------------------
  // Pubblicazione Forze/Coppie nel Frame di BASE
  // ----------------------------------------------------------------------
  geometry_msgs::TwistStamped msg_base;
  msg_base.header.stamp = ros::Time::now();

  // --- FORZE (Linear) ---
  // wrench_base_frame_ indices: 0=Fx, 1=Fy, 2=Fz
  msg_base.twist.linear.x = wrench_base_frame_(0);
  msg_base.twist.linear.y = wrench_base_frame_(1);
  msg_base.twist.linear.z = wrench_base_frame_(2);

  // --- COPPIE (Angular) ---
  // wrench_base_frame_ indices: 3=Tx, 4=Ty, 5=Tz
  msg_base.twist.angular.x = wrench_base_frame_(3);
  msg_base.twist.angular.y = wrench_base_frame_(4);
  msg_base.twist.angular.z = wrench_base_frame_(5);

  // Pubblica il messaggio
  F_base_pub.publish(msg_base);

  // Pose Publisher
  std::array<double, 16> x = model_handle_->getPose(franka::Frame::kEndEffector);
  p.x = x[12];
  p.y = x[13];
  p.z = x[14];
  P_pub.publish(p);
  // Velocity Publisher Giovanni
  //Eigen::Matrix<double, 6, 1> vel_data=jacobian * dq;
  //std::array<double, 6> vel_array;
  //for (int i = 0; i < 6; ++i) {
    //vel_array[i] = vel_data[i];
  //}
  //dp.x = vel_array[0];
  //dp.y = vel_array[1];
  //dp.z = vel_array[2];
  //V_pub.publish(dp);
  
  // Velocity Publisher (TwistStamped)
  Eigen::Matrix<double, 6, 1> vel_data = jacobian * dq;
  
  geometry_msgs::TwistStamped vel_msg;
  vel_msg.header.stamp = ros::Time::now();
  // vel_msg.header.frame_id = "panda_link0"; // Opzionale: se vuoi specificare il frame di riferimento

  // Velocità Lineare (v = J_lin * dq)
  vel_msg.twist.linear.x = vel_data[0];
  vel_msg.twist.linear.y = vel_data[1];
  vel_msg.twist.linear.z = vel_data[2];

  // Velocità Angolare (omega = J_ang * dq)
  vel_msg.twist.angular.x = vel_data[3];
  vel_msg.twist.angular.y = vel_data[4];
  vel_msg.twist.angular.z = vel_data[5];

  V_pub.publish(vel_msg);
  
   double dt = period.toSec();
 // -----------------------------------------------------------
  // CALCOLO ANALITICO + FILTRAGGIO ACCELERAZIONE (/a_tj)
  // -----------------------------------------------------------
  
  // 1. Calcolo dell'accelerazione dei giunti (ddq)
  //    ddq = (dq_curr - dq_prev) / dt
  Eigen::Matrix<double, 7, 1> ddq;
  ddq.setZero();
  if (dt > 1e-5) {
      ddq = (dq - dq_prev_) / dt;
  }
  
  // 2. Formula Cinematica Esatta: a = J * ddq + J_dot * dq
  //    (J_dot l'abbiamo calcolato nel blocco precedente per la legge di controllo)
  Eigen::Matrix<double, 6, 1> acc_raw;
  acc_raw = jacobian * ddq + J_dot * dq;

  // 3. Filtro Passa-Basso (Exponential Moving Average)
  //    Questo è CRUCIALE per l'adattamento parametri in Simulink.
  //    y_k = (1 - alpha) * y_{k-1} + alpha * u_k
  acc_filtered_ = (1.0 - acceleration_filter_gain_) * acc_filtered_ + 
                   acceleration_filter_gain_ * acc_raw;

  // 4. Aggiornamento memoria giunti
  dq_prev_ = dq;

  // 5. Pubblicazione del dato FILTRATO
  geometry_msgs::AccelStamped acc_msg;
  acc_msg.header.stamp = ros::Time::now();
  
  // Lineare
  acc_msg.accel.linear.x = acc_filtered_[0];
  acc_msg.accel.linear.y = acc_filtered_[1];
  acc_msg.accel.linear.z = acc_filtered_[2];

  // Angolare
  acc_msg.accel.angular.x = acc_filtered_[3];
  acc_msg.accel.angular.y = acc_filtered_[4];
  acc_msg.accel.angular.z = acc_filtered_[5];

  A_pub.publish(acc_msg);
  // -----------------------------------------------------------
  
  // -----------------------------------------------------------
  // CALCOLO E PUBBLICAZIONE ENERGIA CINETICA
  // Formula: K = 0.5 * dq' * M * dq
  // -----------------------------------------------------------
  
  // Il risultato di (1x7) * (7x7) * (7x1) è uno scalare (1x1)
  // Eigen permette di estrarlo direttamente come value() o cast implicito se è 1x1
  double kinetic_energy = 0.5 * dq.transpose() * M_q * dq;

  std_msgs::Float64 K_msg;
  K_msg.data = kinetic_energy;
  
  K_pub.publish(K_msg);
  // -----------------------------------------------------------
  
  // Force Publisher  
  std::array<double, 6> F_data;
  /*
  Fext.twist.linear.x = Fee[0];
  Fext.twist.linear.y = Fee[1];
  Fext.twist.linear.z = Fee[2];
  Fext.twist.angular.x = Fee[3];
  Fext.twist.angular.y = Fee[4];
  Fext.twist.angular.z = Fee[5];
  */
  Fext.twist.linear.x = wrench_measured[0];
  Fext.twist.linear.y = wrench_measured[1];
  Fext.twist.linear.z = wrench_measured[2];
  Fext.twist.angular.x = wrench_measured[3];
  Fext.twist.angular.y = wrench_measured[4];
  Fext.twist.angular.z = wrench_measured[5];
  F_pub.publish(Fext);
  
  // -------------------------------------------------------
  // Pubblicazione Forze Stimate dal Robot (Franka Internal)
  // -------------------------------------------------------
  geometry_msgs::TwistStamped F_estimated_msg;

  F_estimated_msg.header.stamp = ros::Time::now(); // È buona prassi mettere il timestamp
  F_estimated_msg.twist.linear.x = Fee[0];
  F_estimated_msg.twist.linear.y = Fee[1];
  F_estimated_msg.twist.linear.z = Fee[2];
  F_estimated_msg.twist.angular.x = Fee[3];
  F_estimated_msg.twist.angular.y = Fee[4];
  F_estimated_msg.twist.angular.z = Fee[5];
  F_estimated_pub.publish(F_estimated_msg);
	  
  // Transformation matrix T computation
  Eigen::Matrix3d R;
  R << x[0], x[4], x[8],
       x[1], x[5], x[9],
       x[2], x[6], x[10];

  Eigen::Vector3d euler_angles = R.eulerAngles(2, 1, 2);
  
  
  Eigen::Matrix3d T;
  T << 0, -std::sin(euler_angles[0]), std::sin(euler_angles[0])*std::sin(euler_angles[1]),
       0,  std::cos(euler_angles[0]), std::sin(euler_angles[0])*std::sin(euler_angles[1]),
       1,  0, std::cos(euler_angles[1]);
  
  Eigen::MatrixXd Ta(6, 6);
  Ta << Eigen::Matrix3d::Identity(), Eigen::MatrixXd::Zero(3,3),
        Eigen::MatrixXd::Zero(3,3), T;
 
  Eigen::MatrixXd Ta_pinv;
  pseudoInverse(Ta, Ta_pinv);

  // compute error to desired pose
  Eigen::Matrix<double, 6, 1> error;
  error.head(3) << position - position_d_; 

  // orientation error
  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  // "difference" quaternion
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  // Transform to base frame
  error.tail(3) << -transform.rotation() * error.tail(3);

  /*
  ROS_INFO("Translation Error: [%.4f, %.4f, %.4f] and Orientation Error: [%.4f, %.4f, %.4f]\n",
            error(0), error(1), error(2), error(3), error(4), error(5) );
  */
  // compute control
  // allocate variables
  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

  // pseudoinverse for nullspace handling
  // kinematic pseuoinverse
  Eigen::MatrixXd jacobian_transpose_pinv;
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);


  //std::array<double, 42> initial_jacobian_array = model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  //Eigen::Map<Eigen::Matrix<double, 6, 7>> initial_jacobian(initial_jacobian_array.data());
  //prev_jacobian = initial_jacobian;

  Eigen::MatrixXd jacobian_pinv;
  pseudoInverse(jacobian, jacobian_pinv);
  
  // --- CALCOLO NUOVO E CORRETTO ---
  //double dt = period.toSec();

  if (dt != 0.0) {
      J_dot = (jacobian - prev_jacobian) / dt;
      prev_jacobian = jacobian;
  } else {
      J_dot.setZero();
  }



  Eigen::Matrix<double, 1, 7> zero_t = Eigen::Matrix<double, 7, 1>::Zero();
  Eigen::Matrix<double, 7,7> ID = Eigen::Matrix<double, 7, 7>::Identity();
  
    // Cartesian PD control with damping ratio = 1
  //tau_task << jacobian.transpose() * (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));

     // CONTROLLO DARIO
     
      tau_task << jacobian.transpose() * (- cartesian_stiffness_ * error  - cartesian_damping_ * (jacobian * dq - velocity_desired) ) + M_q * jacobian_pinv *(acceleration_desired - J_dot * dq); //
     
     /*
     tau_task << M_q * jacobian_pinv * (- cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq - velocity_desired)) + M_q * jacobian_pinv *(acceleration_desired - J_dot * dq); 
     */
     // CONTROLLO GIOVANNI
    /*
    tau_task <<  M_q * jacobian_pinv * cartesian_inertia_inv_ *
    		 (
    		 cartesian_inertia_ * acceleration_desired  -
    		 cartesian_inertia_ * J_dot * dq -
    		 cartesian_stiffness_ * error -
    		 cartesian_damping_ * (jacobian * dq - velocity_desired) +
    		 Ta_pinv.transpose()*Fee
    		 ); 
    */
/* tau_task <<  M_q * jacobian_pinv * (acceleration_desired - J_dot * dq) + jacobian.transpose() * ( - cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq - velocity_desired) ) + M_q * jacobian_pinv * cartesian_inertia_inv_ *Fee; */ //Ta
//5/02/2024   
 /*    jacobian.transpose() * (- cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq - velocity_desired)) + M_q * jacobian_pinv *(acceleration_desired - J_dot * dq);*/
  


  tau_nullspace << ( 
  		     Eigen::MatrixXd::Identity(7, 7) 
                     - jacobian.transpose() * jacobian_transpose_pinv) * (nullspace_stiffness_ * (q_d_nullspace_ - q)
                     - (2.0 * sqrt(nullspace_stiffness_)) * dq
                   ); 
// !!!!!! COMPENSAZIONE QUI !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  Eigen::Matrix<double,7,1> tau_comp;
  tau_comp << 0.7666110992431774, -1.1455045815069342, 0.8158296585093106, 1.1696758984080802, 0.9405113560393308, 0.19704928068995708, 0.00021074612972737938;
  // Desired torque
  tau_d <<  coriolis + tau_nullspace + tau_task + tau_comp; // - jacobian.transpose() * Ta_pinv.transpose()*Fee; //(da usare con ROS_TEST_5) - jacobian.transpose() * Fee + tau_task
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//  tau_d << tau_task + coriolis;

  // Saturate torque rate to avoid discontinuities
  tau_d << saturateTorqueRate(tau_d, tau_J_d);
  for (size_t i = 0; i < 7; ++i) {
    joint_handles_[i].setCommand(tau_d(i));
  }

  // update parameters changed online either through dynamic reconfigure or through the interactive
  // target by filtering
  cartesian_stiffness_   = filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
  cartesian_damping_     = filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
  cartesian_inertia_     = filter_params_ * cartesian_inertia_target_ + (1.0 - filter_params_) * cartesian_inertia_;
  cartesian_inertia_inv_ = filter_params_ * cartesian_inertia_target_inv_ + (1.0 - filter_params_) * cartesian_inertia_inv_;  
  nullspace_stiffness_   = filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;


  std::lock_guard<std::mutex> position_d_target_mutex_lock( position_and_orientation_d_target_mutex_);
  position_d_    = position_d_target_; //filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_target_; //orientation_d_.slerp(filter_params_, orientation_d_target_);

  m.x = cartesian_inertia_(0,0);
  m.y = 0;
  m.z = 0;
  m_pub.publish(m);
//END update
}



//----------------------------------------------------------------------------- 
  Eigen::Matrix<double, 7, 1> CartesianImpedanceExampleController_Dario::saturateTorqueRate(
  const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
  const Eigen::Matrix<double, 7, 1>& tau_J_d)
//-----------------------------------------------------------------------------   
{  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] = tau_d_calculated[i]+ std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
  
//END saturateTorqueRate  
}



//----------------------------------------------------------------------------- 
void CartesianImpedanceExampleController_Dario::complianceParamCallback( franka_example_controllers::compliance_paramConfig& config, uint32_t /*level*/)
//----------------------------------------------------------------------------- 
{
  //------------------------------
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3) << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3) << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  //------------------------------
  cartesian_damping_target_.setIdentity();
  cartesian_damping_target_.topLeftCorner(3, 3) << 0.5 * sqrt( 2.0 * config.translational_stiffness) * Eigen::Matrix3d::Identity(); // *config.translational_inertia
  cartesian_damping_target_.bottomRightCorner(3, 3) << 0.5 * sqrt(2.0 * config.rotational_stiffness) * Eigen::Matrix3d::Identity(); // *config.rotational_inertia
  //------------------------------
  cartesian_inertia_target_.setIdentity();
  cartesian_inertia_target_.topLeftCorner(3, 3) << config.translational_inertia * Eigen::Matrix3d::Identity();
  cartesian_inertia_target_.bottomRightCorner(3, 3) << config.rotational_inertia * Eigen::Matrix3d::Identity();
  //------------------------------  
  cartesian_inertia_target_inv_.setIdentity();
  cartesian_inertia_target_inv_.topLeftCorner(3, 3) <<  Eigen::Matrix3d::Identity() / config.translational_inertia;
  cartesian_inertia_target_inv_.bottomRightCorner(3, 3) <<  Eigen::Matrix3d::Identity() / config.rotational_inertia;
  //------------------------------  
//  ROS_INFO( "Inertia: [%.4f, %.4f,%.4f, %.4f,%.4f, %.4f,%.4f, %.4f,%.4f]\n", config.translational_inertia,config.rotational_inertia);
  ROS_INFO( "Damping: [%.4f,%.4f,%.4f,%.4f, %.4f,%.4f]\n", cartesian_damping_target_(0,0),cartesian_damping_target_(1,1),cartesian_damping_target_(2,2),cartesian_damping_target_(3,3),cartesian_damping_target_(4,4),cartesian_damping_target_(5,5));
  nullspace_stiffness_target_ = config.nullspace_stiffness;
  

//END complianceParamCallback  
}

//----------------------------------------------------------------------------- 
void CartesianImpedanceExampleController_Dario::equilibriumPoseCallback( const geometry_msgs::PoseStamped& msg)
//----------------------------------------------------------------------------- 
{

// [FIX] Controllo rapido anti-NaN sulla posizione
  if (std::isnan(msg.pose.position.x) || std::isnan(msg.pose.position.y) || std::isnan(msg.pose.position.z)) {
       ROS_WARN_THROTTLE(1.0, "ATTENZIONE: Ricevuto NaN su /equilibrium_pose! Ignorato.");
       return;
  }
  std::lock_guard<std::mutex> position_d_target_mutex_lock(
      position_and_orientation_d_target_mutex_);
      
  position_d_target_ << msg.pose.position.x, msg.pose.position.y, msg.pose.position.z;
  
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  
  orientation_d_target_.coeffs() << msg.pose.orientation.x, msg.pose.orientation.y,
      msg.pose.orientation.z, msg.pose.orientation.w;
      
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }
  
/*
  ROS_INFO("Linear Position: [%.4f, %.4f, %.4f], Angular Position: [%.4f, %.4f, %.4f, %.4f]\n",
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w ); 
*/
//  orientation_d_target_.coeffs() << orientation_d_target_0.coeffs();
//END equilibriumPoseCallback  
}

//--------forcesensorcallback vecchia di giovanni------------------------- 
//void CartesianImpedanceExampleController_Dario::forcesensorCallback( const geometry_msgs::WrenchStamped& msg)
//----------------------------------------------------------------------------- 
//{
  //double Fx = msg.wrench.force.x;
  //double Fy = msg.wrench.force.y;
  //double Fz = msg.wrench.force.z;

  //double Tx = msg.wrench.torque.x;
  //double Ty = msg.wrench.torque.y;
  //double Tz = msg.wrench.torque.z;
  
  //std::lock_guard<std::mutex> wrench_lock(wrench_measured_mutex);
  
  //wrench_measured << Fx, Fy, Fz, Tx, Ty, Tz;

//  ROS_INFO( "Measured Force: [%.4f, %.4f, %.4f], Measured Torque: [%.4f, %.4f, %.4f]\n", Fx, Fy, Fz, Tx, Ty, Tz);
//END forcesensorCallback  
//}
//---------forcesensorcallback for soft bias-------------------------------- 
void CartesianImpedanceExampleController_Dario::forcesensorCallback(const geometry_msgs::WrenchStamped& msg)
//----------------------------------------------------------------------------- 
{
  // 1. Estrazione dei dati grezzi dal messaggio ROS
  double Fx = msg.wrench.force.x;
  double Fy = msg.wrench.force.y;
  double Fz = msg.wrench.force.z;
  double Tx = msg.wrench.torque.x;
  double Ty = msg.wrench.torque.y;
  double Tz = msg.wrench.torque.z;
  
  // Creiamo un vettore temporaneo con i dati grezzi
  Eigen::Matrix<double, 6, 1> raw_wrench;
  raw_wrench << Fx, Fy, Fz, Tx, Ty, Tz;

  // Proteggiamo l'accesso alle variabili condivise
  std::lock_guard<std::mutex> wrench_lock(wrench_measured_mutex);

  // 2. LOGICA SOFT BIAS:
  // Se è il primo pacchetto che riceviamo, lo salviamo come offset (tara)
  if (!is_bias_initialized_) {
      wrench_bias_ = raw_wrench;
      is_bias_initialized_ = true;
      ROS_INFO("Sensore ATI: Bias acquisito. Offset applicato: F=[%.2f, %.2f, %.2f]", Fx, Fy, Fz);
  }

  // 3. Calcolo del valore netto (Misura Attuale - Bias Iniziale)
  wrench_measured = raw_wrench - wrench_bias_;

  // Debug opzionale (puoi commentarlo se intasa il terminale)
  // ROS_INFO_THROTTLE(1.0, "Force Netta Z: %.2f (Grezza: %.2f, Bias: %.2f)", 
  //                   wrench_measured[2], raw_wrench[2], wrench_bias_[2]);
}


//-----------------------------------------------------------------------------    
void CartesianImpedanceExampleController_Dario::DesiredVelocitySubscriberCallback(
    const geometry_msgs::TwistStamped& msg) {
//-----------------------------------------------------------------------------    
  // Extracting the linear velocity data
  double vx = msg.twist.linear.x;
  double vy = msg.twist.linear.y;
  double vz = msg.twist.linear.z;

  // Extracting the angular velocity data
  double wx = msg.twist.angular.x;
  double wy = msg.twist.angular.y;
  double wz = msg.twist.angular.z;
  
  // [FIX] PROTEZIONE NAN: Se uno dei valori non è un numero, ignora tutto il pacchetto
  if (std::isnan(vx) || std::isnan(vy) || std::isnan(vz) || 
      std::isnan(wx) || std::isnan(wy) || std::isnan(wz)) {
      ROS_WARN_THROTTLE(1.0, "ATTENZIONE: Ricevuto NaN su /desired_velocity! Pacchetto ignorato.");
      return; 
  }

 // Protect the member variable with a mutex
  std::lock_guard<std::mutex> lock(velocity_desired_mutex);
  
  // Assigning the velocities to the desired_velocity member variable
  velocity_desired << vx, vy, vz, wx, wy, wz;

  // Print the received velocity data
//  ROS_INFO("Linear Velocity: [%.4f, %.4f, %.4f], Angular Velocity: [%.4f, %.4f, %.4f]\n", vx, vy, vz, wx, wy, wz);
}

//-----------------------------------------------------------------------------    
void CartesianImpedanceExampleController_Dario::DesiredAccelerationSubscriberCallback(
    const geometry_msgs::AccelStamped& msg) {
//-----------------------------------------------------------------------------    
  // Extracting the linear acceleration data
  double ax = msg.accel.linear.x;
  double ay = msg.accel.linear.y;
  double az = msg.accel.linear.z;

  // Extracting the angular acceleration data
  double awx = msg.accel.angular.x;
  double awy = msg.accel.angular.y;
  double awz = msg.accel.angular.z;
  
  // [FIX] PROTEZIONE NAN
  if (std::isnan(ax) || std::isnan(ay) || std::isnan(az) || 
      std::isnan(awx) || std::isnan(awy) || std::isnan(awz)) {
      ROS_WARN_THROTTLE(1.0, "ATTENZIONE: Ricevuto NaN su /desired_acceleration! Pacchetto ignorato.");
      return;
  }

  // Protect the member variable with a mutex
  std::lock_guard<std::mutex> lock(acceleration_desired_mutex);
  
  // Assigning the accelerations to the acceleration_desired member variable
  acceleration_desired << ax, ay, az, awx, awy, awz;

  // Print the received acceleration data
  // ROS_INFO("Linear Acceleration: [%.4f, %.4f, %.4f], Angular Acceleration: [%.4f, %.4f, %.4f]\n", ax, ay, az, awx, awy, awz);
}

void CartesianImpedanceExampleController_Dario::resetBiasCallback(const std_msgs::Bool& msg) {
    if (msg.data) {
        std::lock_guard<std::mutex> wrench_lock(wrench_measured_mutex);
        is_bias_initialized_ = false; // Forza il ricalcolo del bias al prossimo update
        ROS_INFO("COMMAND RECEIVED: Bias Reset.");
    }
}

}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::CartesianImpedanceExampleController_Dario,
                       controller_interface::ControllerBase)