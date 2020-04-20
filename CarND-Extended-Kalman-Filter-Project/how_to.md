##Project of EKF setup

### how to programming 
#### kalman_filter module
  1. As you can see, i added predict and update through 25 to 84;
  2. To avoided code repeat, i set the fucntion of `void UpdateRoutine(const Eigen::VectorXd& y);`, which is located at line of 67 in kalman_filter.h.

#### tools module
  1. It is easy to setup based on the fomula of RMSE and polar to Cartesian.
  2. In additon, the dividend neat to zero, i set threshold of 0.0001.
  
#### FusionEKF module
  1. First, i set Lidar and initializtion, which includes H, H_jacobian, P, F, Q, x, and D(a)(acceleration noixse matrix).
  2. Next, i initialized the data based on the sensor type, which are rho, phi, roh_dot or px,py.
  3. In the section of prediction, the process noise covariance matrix of Q is updated based on realtime. for initializing Q, i set noisy_a<<9,9, which are acceleration parameters, more details are shown following:
  ```principle code
  because of Q << D(p),      cov(p,v),
                  cov(p,v),  D(a);
  v = a*t + v0;
  p = v0*t + 1/2*a*t^2;
  D(v) = D(a*t) = t^2*D(a);
  D(p) = D(1/2*a*t**2) = 1/4*t^4*D(a);
  cov(p,v) = sqrt(D(v))*sqrt(D(p)) = 1/2*t^3*D(a);
  ```
  4. Last, for the update, i use the sensor type to perform the update step. and update the state and covariance matrices.
#### Result 
  . I push results data of the x and P into the file name of `result_x_p.txt`.
  . RMSE value:[0.0973,0.0855,0.4513,0.4399].
  
#### resubmit statement
. After first submit, i found accuracy of rmse is not met requirement. then i added timestamp of acc in the line of 96 and 105, that is `previous_timestamp_ = measurement_pack.timestamp_;`, but i can not resubmit project after 15min, but it will not change the submission under review. Forgive my carelessness.
. Following your advice, acc can be set like pics, that a good idea, thanks your advice.

