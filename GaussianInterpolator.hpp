#include "src/brownian-motion/2DBrownianMotionPath.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits>
#include "src/nlopt/api/nlopt.hpp"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

double gp_logit(double p);
double gp_logit_inv(double r);

struct likelihood_point_transformed;

struct parameters_nominal {
  double sigma_2;
  //
  double phi;
  //
  int nu;
  double tau_2;
  std::vector<double> lower_triag_mat_as_vec = std::vector<double> (28, 1.0);

  parameters_nominal()
    : sigma_2(0.0001),
      phi(-10.0),
      nu(5),
      tau_2(1.0)
  {
    lower_triag_mat_as_vec = std::vector<double> {0.1, //0
						  0.0, 0.1, //2
						  0.0, 0.0, 0.1, //5
						  0.0, 0.0, 0.0, 0.1, //9
						  0.0, 0.0, 0.0, 0.0, 0.1, //14
						  0.0, 0.0, 0.0, 0.0, 0.0, 0.1, //20
						  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1}; //27
  };

  parameters_nominal(const std::vector<double> &x)
    : sigma_2(x[0]),
      phi(x[1]),
      nu(x[2]),
      tau_2(x[3]),
      lower_triag_mat_as_vec(std::vector<double>(28, 1.0))
  {
    for (unsigned i=0; i<28; ++i) {
      lower_triag_mat_as_vec[i] = x[i+4];
    }
  }

  parameters_nominal& operator=(const parameters_nominal& rhs)
  {
    if (this==&rhs) {
      return *this;
    } else {
      sigma_2 = rhs.sigma_2;
      phi = rhs.phi;
      nu = rhs.nu;
      tau_2 = rhs.tau_2;
      lower_triag_mat_as_vec = rhs.lower_triag_mat_as_vec;

      return *this;
    }
  }

  parameters_nominal(const parameters_nominal& rhs)
  {
    sigma_2 = rhs.sigma_2;
    phi = rhs.phi;
    nu = rhs.nu;
    tau_2 = rhs.tau_2;
    lower_triag_mat_as_vec = rhs.lower_triag_mat_as_vec;
  }

  friend std::ostream& operator<<(std::ostream& os,
				  const parameters_nominal& current_parameters)
  {
    os << "sigma_2=" << std::scientific << std::setprecision(14) << current_parameters.sigma_2 << ";\n";
    os << "phi=" << std::scientific << std::setprecision(14) << current_parameters.phi << ";\n";
    //
    os << "nu=" << std::scientific << std::setprecision(14) << current_parameters.nu << ";\n";
    os << "tau_2=" << std::scientific << std::setprecision(14) << current_parameters.tau_2 << ";\n";
    //
    unsigned counter = 0;
    for (unsigned i=0; i<7; ++i) {
      for (unsigned j=0; j<=i; ++j) {
	if (j==i) {
	  os << std::scientific << std::setprecision(14) << current_parameters.lower_triag_mat_as_vec[counter] << ";\n";
	} else {
	  os << std::scientific << std::setprecision(14) << current_parameters.lower_triag_mat_as_vec[counter] << ",";
	}
	counter = counter + 1;
      }
    }
    return os;
  }

  friend std::istream& operator>>(std::istream& is,
				  parameters_nominal& target_parameters)
  {
    std::string name;
    std::string value;
    // sigma_2
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    target_parameters.sigma_2 = std::stod(value);

    // phi
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    target_parameters.phi = std::stod(value);

    // nu
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    target_parameters.nu = std::stod(value);

    // tau_2
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    target_parameters.tau_2 = std::stod(value);

    // L
    unsigned counter = 0;
    for (unsigned i=0; i<7; ++i) {
      for (unsigned j=0; j<=i; ++j) {

	if (j==i) {
	  std::getline(is, value, ';');
	  target_parameters.lower_triag_mat_as_vec[counter] = std::stod(value);
	} else {
	  std::getline(is, value, ',');
	  target_parameters.lower_triag_mat_as_vec[counter] = std::stod(value);
	}
	counter = counter + 1;
      }
    }

    return is;
  }

  std::vector<double> as_vector() const
  {
    std::vector<double> out = std::vector<double> (32);
    out[0] = sigma_2;
    out[1] = phi;
    out[2] = nu;
    out[3] = tau_2;

    for (unsigned i=4; i<32; ++i) {
      out[i] = lower_triag_mat_as_vec[i-4];
    }

    return out;
  }

  gsl_matrix* lower_triag_matrix() const
  {
    gsl_matrix* out = gsl_matrix_calloc(7,7);
    unsigned counter = 0;

    for (unsigned i=0; i<7; ++i) {
      for (unsigned j=0; j<=i; ++j) {
	gsl_matrix_set(out, i, j,
		       lower_triag_mat_as_vec[counter]);
	counter = counter + 1;
      }
    }

    return out;
  }

};


struct likelihood_point {
  double x_0_tilde;
  double y_0_tilde;
  //
  double x_t_tilde;
  double y_t_tilde;
  //
  double sigma_y_tilde;
  double t_tilde;
  double rho;
  //
  double log_likelihood;
  //
  bool FLIPPED;

  likelihood_point()
    : x_0_tilde(0.5),
      y_0_tilde(0.5),
      x_t_tilde(0.5),
      y_t_tilde(0.5),
      sigma_y_tilde(1.0),
      t_tilde(1.0),
      rho(0.0),
      log_likelihood(1.0),
      FLIPPED(false)
  {}

  likelihood_point(const likelihood_point& current_lp) 
  {
    x_0_tilde = current_lp.x_0_tilde;
    y_0_tilde = current_lp.y_0_tilde;
    //
    x_t_tilde = current_lp.x_t_tilde;
    y_t_tilde = current_lp.y_t_tilde;
    //
    sigma_y_tilde = current_lp.sigma_y_tilde;
    t_tilde = current_lp.t_tilde;
    rho = current_lp.rho;
    //
    log_likelihood = current_lp.log_likelihood;
    //
    FLIPPED = current_lp.FLIPPED;
  }

  likelihood_point(double x_0_tilde_in,
		   double y_0_tilde_in,
		   double x_t_tilde_in,
		   double y_t_tilde_in,
		   double sigma_y_tilde_in,
		   double t_tilde_in,
		   double rho_in,
		   double log_likelihood_in,
		   bool FLIPPED_IN)
    : x_0_tilde(x_0_tilde_in),
      y_0_tilde(y_0_tilde_in),
      x_t_tilde(x_t_tilde_in),
      y_t_tilde(y_t_tilde_in),
      sigma_y_tilde(sigma_y_tilde_in),
      t_tilde(t_tilde_in),
      rho(rho_in),
      log_likelihood(log_likelihood_in),
      FLIPPED(FLIPPED_IN)
  {}

  likelihood_point(double rho_in,
		   double sigma_x,
		   double sigma_y,
		   double x_0_in,
		   double y_0_in,
		   double x_t,
		   double y_t,
		   double t,
		   double a,
		   double b,
		   double c,
		   double d) 
  {
    double Lx = b - a;
    double Ly = d - c;

    double x_T = x_t - a;
    double x_0 = x_0_in - a;

    double y_T = y_t - c;
    double y_0 = y_0_in - c;
    // STEP 1
    double tau_x = sigma_x/Lx;
    double tau_y = sigma_y/Ly;

    x_t_tilde = x_T/Lx;
    y_t_tilde = y_T/Ly;
    x_0_tilde = x_0/Lx;
    y_0_tilde = y_0/Ly;
    sigma_y_tilde = tau_y/tau_x;
    t_tilde = t*std::pow(tau_x, 2);
    FLIPPED = false;
	
    if (tau_x < tau_y) {
      FLIPPED = true;
      x_t_tilde = y_T/Ly;
      y_t_tilde = x_T/Lx;
      //
      x_0_tilde = y_0/Ly;
      y_0_tilde = x_0/Lx;
      //
      sigma_y_tilde = tau_x/tau_y;
      //
      t_tilde = t*std::pow(tau_y, 2);
    }

    rho = rho_in;
    log_likelihood = 1.0;
  }

  likelihood_point& operator=(const likelihood_point& rhs)
  {
    if (this==&rhs) {
      return *this;
    } else {
      x_0_tilde = rhs.x_0_tilde;
      y_0_tilde = rhs.y_0_tilde;
      //
      x_t_tilde = rhs.x_t_tilde;
      y_t_tilde = rhs.y_t_tilde;
      //
      sigma_y_tilde = rhs.sigma_y_tilde;
      t_tilde = rhs.t_tilde;
      rho = rhs.rho;
      //
      log_likelihood = rhs.log_likelihood;
      FLIPPED = rhs.FLIPPED;

      return *this;
    }
  }

  likelihood_point& operator=(const likelihood_point_transformed& rhs);

  likelihood_point& operator=(const BrownianMotion& BM)
  {
    double Lx = BM.get_b() - BM.get_a();
    double Ly = BM.get_d() - BM.get_c();

    double x_T = BM.get_x_T() - BM.get_a();
    double x_0 = BM.get_x_0() - BM.get_a();

    double y_T = BM.get_y_T() - BM.get_c();
    double y_0 = BM.get_y_0() - BM.get_c();
    double t = BM.get_t();
    // STEP 1
    double tau_x = BM.get_sigma_x()/Lx;
    double tau_y = BM.get_sigma_y()/Ly;

    x_t_tilde = x_T/Lx;
    y_t_tilde = y_T/Ly;
    x_0_tilde = x_0/Lx;
    y_0_tilde = y_0/Ly;
    sigma_y_tilde = tau_y/tau_x;
    t_tilde = std::pow(tau_x, 2);
    FLIPPED = false;
	
    if (tau_x < tau_y) {
      FLIPPED = true;
      x_t_tilde = y_T/Ly;
      y_t_tilde = x_T/Lx;
      //
      x_0_tilde = y_0/Ly;
      y_0_tilde = x_0/Lx;
      //
      sigma_y_tilde = tau_x/tau_y;
      //
      t_tilde = t*std::pow(tau_y, 2);
    }

    rho = BM.get_rho();
    log_likelihood = 1.0;
    
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os,
				  const likelihood_point& point)
  {
    os << "x_0_tilde=" << std::scientific << std::setprecision(14) << point.x_0_tilde << ";\n";
    os << "y_0_tilde=" << std::scientific << std::setprecision(14) << point.y_0_tilde << ";\n";
    //
    os << "x_t_tilde=" << std::scientific << std::setprecision(14) << point.x_t_tilde << ";\n";
    os << "y_t_tilde=" << std::scientific << std::setprecision(14) << point.y_t_tilde << ";\n";
    //
    os << "sigma_y_tilde=" << std::scientific << std::setprecision(14) << point.sigma_y_tilde << ";\n";
    os << "t_tilde=" << std::scientific << std::setprecision(14) << point.t_tilde << ";\n";
    os << "rho=" << std::scientific << std::setprecision(14) << point.rho << ";\n";
    //
    os << "log_likelihood=" << std::scientific << std::setprecision(14) << point.log_likelihood << ";\n";
    os << "FLIPPED=" << std::scientific << std::setprecision(14) << point.FLIPPED << ";\n";

    return os;
  }

  void print_point()
  {
    printf("x_0_tilde=%f \ny_0_tilde=%f \nx_t_tilde=%f \ny_t_tilde=%f \nsigma_y_tilde=%f \nt_tilde=%f \nrho=%f \nlog_likelihood=%f \nFLIPPED=%d \n",
  	   x_0_tilde,
  	   y_0_tilde,
  	   x_t_tilde,
  	   y_t_tilde,
  	   sigma_y_tilde,
  	   t_tilde,
  	   rho,
  	   log_likelihood,
	   FLIPPED);
  }

  friend std::istream& operator>>(std::istream& is,
				  likelihood_point& point)
  {
    std::string name;
    std::string value;
    // x_0_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.x_0_tilde = std::stod(value);

    // y_0_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.y_0_tilde = std::stod(value);

    // x_t_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.x_t_tilde = std::stod(value);

    // y_t_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.y_t_tilde = std::stod(value);

    // sigma_y_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.sigma_y_tilde = std::stod(value);

    // t_tilde
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.t_tilde = std::stod(value);

    // rho
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.rho = std::stod(value);

    // log_likelihood
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.log_likelihood = std::stod(value);

    // FLIPPED
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.FLIPPED = std::stoi(value);

    return is;
  }

  gsl_vector* as_gsl_vector() const
  {
    gsl_vector* out = gsl_vector_alloc(7);
    gsl_vector_set(out, 0, x_0_tilde);
    gsl_vector_set(out, 1, y_0_tilde);
    //
    gsl_vector_set(out, 2, x_t_tilde);
    gsl_vector_set(out, 3, y_t_tilde);
    //
    gsl_vector_set(out, 4, sigma_y_tilde);
    gsl_vector_set(out, 5, t_tilde);
    gsl_vector_set(out, 6, rho);

    return out;
  }
};

struct likelihood_point_transformed {
  double x_0_tilde_transformed;
  double y_0_tilde_transformed;
  //
  double x_t_tilde_transformed;
  double y_t_tilde_transformed;
  //
  double sigma_y_tilde_transformed;
  double t_tilde_transformed;
  double rho_transformed;
  //
  double log_likelihood_transformed;
  bool FLIPPED;
  //
  //
  void set_likelihood_point_transformed(const likelihood_point& lp);
  likelihood_point_transformed(const likelihood_point& lp);
  likelihood_point_transformed();
  //
  gsl_vector* as_gsl_vector() const;

  friend std::ostream& operator<<(std::ostream& os,
				  const likelihood_point_transformed& point);
};

double likelihood_point_distance(const likelihood_point& lp1,
				 const likelihood_point& lp2,
				 const parameters_nominal& params);


double covariance(const likelihood_point& lp1,
		  const likelihood_point& lp2,
		  const parameters_nominal& params);

gsl_matrix* covariance_matrix(const std::vector<likelihood_point>& lps,
			      const parameters_nominal& params);


struct GaussianInterpolator {
  std::vector<likelihood_point_transformed> points_for_integration;
  std::vector<likelihood_point_transformed> points_for_interpolation;
  parameters_nominal parameters;
  gsl_matrix* C;
  gsl_matrix* Cinv;
  gsl_vector* y;
  gsl_vector* mean;
  gsl_vector* difference;
  gsl_vector* c;

  GaussianInterpolator();
  GaussianInterpolator(const std::vector<likelihood_point>& points_for_integration_in,
		       const std::vector<likelihood_point>& points_for_interpolation_in,
		       const parameters_nominal& params);
  GaussianInterpolator(const std::vector<likelihood_point>& points_for_integration_in,
		       const std::vector<likelihood_point>& points_for_interpolation_in);
  
  GaussianInterpolator(const GaussianInterpolator& rhs);
  
  virtual GaussianInterpolator& operator=(const GaussianInterpolator& rhs);

  virtual ~GaussianInterpolator() {
    gsl_matrix_free(C);
    gsl_matrix_free(Cinv);
    gsl_vector_free(y);
    gsl_vector_free(mean);
    gsl_vector_free(difference);
    gsl_vector_free(c);
  }

  inline void set_parameters(const parameters_nominal& new_parameters) {
    parameters = new_parameters;
    set_linalg_members();
  }
  
  void set_linalg_members();
  inline void add_point_for_integration(const likelihood_point_transformed& new_point) {
    points_for_integration.push_back(new_point);
  }
  inline void add_point_for_interpolation(const likelihood_point_transformed& new_point) {
    points_for_interpolation.push_back(new_point);
    set_linalg_members();
  }

  gsl_matrix* covariance_matrix(const std::vector<likelihood_point_transformed>& lps,
				const parameters_nominal& params) const;

  double covariance(const likelihood_point_transformed& lp1,
		    const likelihood_point_transformed& lp2,
		    const parameters_nominal& params) const;

  double likelihood_point_distance(const likelihood_point_transformed& lp1,
				   const likelihood_point_transformed& lp2,
				   const parameters_nominal& params) const;

  double operator()(const likelihood_point& x);
  double prediction_standard_dev(const likelihood_point& x);
  static double optimization_wrapper(const std::vector<double> &x,
				     std::vector<double> &grad,
				     void * data);
  void optimize_parameters();
};

struct PointChecker {
  gsl_vector * mu_mean;
  gsl_matrix * sigma_cov;
  gsl_matrix * sigma_cov_inv;

  PointChecker();
  PointChecker(const PointChecker& rhs);
  PointChecker& operator=(const PointChecker& rhs);

  ~PointChecker() {
    gsl_vector_free(mu_mean);
    gsl_matrix_free(sigma_cov);
    gsl_matrix_free(sigma_cov_inv);
  }
  PointChecker(const std::vector<likelihood_point_transformed>& lptrs);
  double mahalanobis_distance(const likelihood_point& lp) const;

  friend std::ostream& operator<<(std::ostream& os,
				  const PointChecker& checker);
};

struct GaussianInterpolatorWithChecker : 
  public GaussianInterpolator {
  PointChecker point_checker;
  
  GaussianInterpolatorWithChecker();
  GaussianInterpolatorWithChecker(const std::vector<likelihood_point>& points_for_integration_in,
				  const std::vector<likelihood_point>& points_for_interpolation_in,
				  const parameters_nominal& params);
  GaussianInterpolatorWithChecker(const std::vector<likelihood_point>& points_for_integration_in,
				  const std::vector<likelihood_point>& points_for_interpolation_in);

  GaussianInterpolatorWithChecker(const GaussianInterpolatorWithChecker& rhs);
  // GaussianInterpolatorWithChecker(const GaussianInterpolator& rhs);
  virtual ~GaussianInterpolatorWithChecker() {}

  GaussianInterpolatorWithChecker& operator=(const GaussianInterpolatorWithChecker& rhs);
  double check_point(const likelihood_point& lp) const;
};
