#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {

  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = 7;

  // initial state vector
  x_ = VectorXd(n_x_);
  x_.fill(0);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);
  P_.fill(0);

  time_us_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.75;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // weights for the sigmapoints
  weights_ = VectorXd(2*n_aug_+1);

  // Prediction Sigmapoints
  Xsig_pred_ = MatrixXd(n_x_, 2* n_aug_ +1); //n_x_ first

  // lambda-value
  lambda_ = 3 - n_aug_;

  // NIS-values for radar and laser
  NIS_laser_ = 0;

  NIS_radar_ = 0;

  NIS_laser_amount = 0;

  NIS_radar_amount = 0;

  NIS = 0;
}

UKF::~UKF() {}

// Process Measurement function
void UKF::ProcessMeasurement(MeasurementPackage meas_pack) {

  //checking if ukf is initialized already
  if (!is_initialized_) {

    // Initialize the ukf
    x_ << 1, 1, 1, 1, 1;

    P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;


    if (meas_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
      x_[0] = meas_pack.raw_measurements_[0]*cos(meas_pack.raw_measurements_[1]);
      x_[1] = meas_pack.raw_measurements_[0]*sin(meas_pack.raw_measurements_[1]);
      x_[2] = 0;
      x_[3] = 0;
      x_[4] = 0;
    }

    else if (meas_pack.sensor_type_ == MeasurementPackage::LASER)
    {
      x_[0] = meas_pack.raw_measurements_[0];
      x_[1] = meas_pack.raw_measurements_[1];
      x_[2] = 0;
      x_[3] = 0;
      x_[4] = 0;
    }

    // Checking for init-values around 0
    if (fabs(x_[0]) <= 0.001 && fabs(x_[1]) <= 0.001)
    {
      x_[0] = 0.001;
      x_[1] = 0.001;
    }

  // Setting time
  time_us_ = meas_pack.timestamp_;

  is_initialized_ = true;
  return;
  }

  // Compute difference of time
  double dt = (meas_pack.timestamp_ - time_us_) / 1000000.0;
  time_us_ = meas_pack.timestamp_;

  // Predict
  Prediction(dt);

  // Update
  if (meas_pack.sensor_type_ == MeasurementPackage::RADAR)
  {
    UpdateRadar(meas_pack);
  }
  else if (meas_pack.sensor_type_ == MeasurementPackage::LASER)
  {
    UpdateLidar(meas_pack);
  }

}

// Ukf-Prediction
void UKF::Prediction(double delta_t) {

  // Setting augmented variables
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

  VectorXd x_aug = VectorXd(7);

  MatrixXd P_aug = MatrixXd(7, 7);

  Xsig_aug.fill(0);

  // Filling in values
  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_+1) = 0;

  P_aug.fill(0);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(5, 5) = pow(std_a_, 2);
  P_aug(6, 6) = pow(std_yawdd_, 2);

  MatrixXd L = P_aug.llt().matrixL();

  Xsig_aug.col(0) = x_aug;

  // Compute augmented Sigmapoints
  for (int i = 0; i < n_aug_; i++)
  {
      Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*L.col(i);
      Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_)*L.col(i);
  }

  for (int i = 0; i < 2*n_aug_+1; i++)
  {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double yaw = Xsig_aug(3, i);
    double yawd = Xsig_aug(4, i);
    double nu_a = Xsig_aug(5, i);
    double nu_yawdd = Xsig_aug(6, i);

    double px_p, py_p;

    if (fabs(yawd) > 0.001)
    {
      px_p = p_x + (v/yawd) * ( sin( yaw + yawd * delta_t) - sin(yaw));
      py_p = p_y + (v/yawd) * ( cos(yaw) - cos( yaw + yawd * delta_t ));
    }
    else
    {
      px_p = p_x + v*delta_t*cos(yaw);
      py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    px_p = px_p + 0.5*nu_a*pow(delta_t, 2) * cos(yaw);
    py_p = py_p + 0.5*nu_a*pow(delta_t, 2) * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*pow(delta_t, 2);
    yawd_p = yawd_p + nu_yawdd*delta_t;

    // Updating to Prediction Sigmapoints
    Xsig_pred_(0, i) = px_p;
    Xsig_pred_(1, i) = py_p;
    Xsig_pred_(2, i) = v_p;
    Xsig_pred_(3, i) = yaw_p;
    Xsig_pred_(4, i) = yawd_p;

  }

  double weight_0 = lambda_/(lambda_+n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i<2*n_aug_+1; i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  VectorXd x_pred_ = VectorXd(n_x_);

  MatrixXd P_pred_ = MatrixXd(n_x_, n_x_);

  // Predict mean x_pred_ and Covariancematrix  P_pred_
  x_pred_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x_pred_ = x_pred_ + weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  P_pred_.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_pred_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;


    P_pred_ = P_pred_ + weights_(i) * x_diff * x_diff.transpose();
  }

  x_ = x_pred_;
  P_ = P_pred_;
}


// Ukf-update for Laser
void UKF::UpdateLidar(MeasurementPackage meas_package) {

  int n_z = 2;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Change dimension to measurement space
  for  (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v_x = cos(yaw)*v;
    double v_y = sin(yaw)*v;

    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;
  }

  // make Prediction for update step
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);

  for (int i=0; i < 2*n_aug_+1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // Compute R
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_*std_laspx_, 0,
          0, std_laspy_*std_laspy_;

  S = S + R;

  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // Compute TC-Matrix
  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();

  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Compute NIS-values
  NIS = z_diff.transpose() * S.inverse() * z_diff;

  NIS_laser_amount = NIS_laser_amount +1;

  if (NIS > 5.991)
  {
      NIS_laser_ = NIS_laser_ +1;
  }
}

// Ukf-Update for radar
void UKF::UpdateRadar(MeasurementPackage meas_package) {

  int n_z = 3;
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // Change dimension to measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);
    Zsig(1,i) = atan2(p_y,p_x);
    Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);
    }

  // make Prediction for update step
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0);

  for (int i=0; i < 2*n_aug_+1; i++)
  {
    z_pred = z_pred + weights_(i) * Zsig.col(i);
  }

  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++)
  {
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  // compute R
  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_*std_radr_, 0, 0,
          0, std_radphi_*std_radphi_, 0,
          0, 0,std_radrd_*std_radrd_;
  S = S + R;


  // Compute Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  Tc.fill(0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Compute NIS-values
  NIS = z_diff.transpose() * S.inverse() * z_diff;

  NIS_radar_amount = NIS_radar_amount +1;

  if (NIS > 7.815)
  {
      NIS_radar_ = NIS_radar_ +1;
  }

}
