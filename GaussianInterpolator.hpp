#include "2DBrownianMotionPath.hpp"
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
#include <sstream>
#include <limits>
#include "nlopt.hpp"
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <vector>

struct parameters_nominal {
  double sigma_2;
  //
  double phi;
  //
  int nu;
  double tau_2;
  std::vector<double> lower_triag_mat_as_vec = std::vector<double> (28, 1.0);

  parameters_nominal()
    : sigma_2(1.0),
      phi(0.0),
      nu(5),
      tau_2(1.0)
  {
    lower_triag_mat_as_vec = std::vector<double> {1.0, //0
						  0.0, 1.0, //2
						  0.0, 0.0, 1.0, //5
						  0.0, 0.0, 0.0, 1.0, //9
						  0.0, 0.0, 0.0, 0.0, 1.0, //14
						  0.0, 0.0, 0.0, 0.0, 0.0, 1.0, //20
						  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}; //27
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
    os << "sigma_2=" << current_parameters.sigma_2 << ";\n";
    os << "phi=" << current_parameters.phi << ";\n";
    //
    os << "nu=" << current_parameters.nu << ";\n";
    os << "tau_2=" << current_parameters.tau_2 << ";\n";
    //
    unsigned counter = 0;
    for (unsigned i=0; i<7; ++i) {
      for (unsigned j=0; j<=i; ++j) {
	os << current_parameters.lower_triag_mat_as_vec[counter] << " ";
	counter = counter + 1;
      }
      os << std::endl;
    }
    return os;
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
  double likelihood;

  likelihood_point()
    : x_0_tilde(0.5),
      y_0_tilde(0.5),
      x_t_tilde(0.5),
      y_t_tilde(0.5),
      sigma_y_tilde(1.0),
      t_tilde(1.0),
      rho(0.0),
      likelihood(1.0)
  {}

  likelihood_point(double x_0_tilde_in,
		   double y_0_tilde_in,
		   double x_t_tilde_in,
		   double y_t_tilde_in,
		   double sigma_y_tilde_in,
		   double t_tilde_in,
		   double rho_in,
		   double likelihood_in)
    : x_0_tilde(x_0_tilde_in),
      y_0_tilde(y_0_tilde_in),
      x_t_tilde(x_t_tilde_in),
      y_t_tilde(y_t_tilde_in),
      sigma_y_tilde(sigma_y_tilde_in),
      t_tilde(t_tilde_in),
      rho(rho_in),
      likelihood(likelihood_in)
  {}

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
      likelihood = rhs.likelihood;

      return *this;
    }
  }

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
	
    if (tau_x < tau_y) {
      x_t_tilde = y_T/Ly;
      y_t_tilde = x_T/Lx;
      //
      x_0_tilde = y_0/Ly;
      y_0_tilde = x_0/Lx;
      //
      sigma_y_tilde = tau_x/tau_y;
      //
      t_tilde = std::pow(tau_y, 2);
    }

    rho = BM.get_rho();
    likelihood = 0.0;
    
    return *this;
  }

  friend std::ostream& operator<<(std::ostream& os,
				  const likelihood_point& point)
  {
    os << "x_0_tilde=" << point.x_0_tilde << ";\n";
    os << "y_0_tilde=" << point.y_0_tilde << ";\n";
    //
    os << "x_t_tilde=" << point.x_t_tilde << ";\n";
    os << "y_t_tilde=" << point.y_t_tilde << ";\n";
    //
    os << "sigma_y_tilde=" << point.sigma_y_tilde << ";\n";
    os << "t_tilde=" << point.t_tilde << ";\n";
    os << "rho=" << point.rho << ";\n";
    //
    os << "likelihood=" << point.likelihood << ";\n";

    return os;
  }

  void print_point()
  {
    printf("x_0_tilde=%f \ny_0_tilde=%f \nx_t_tilde=%f \ny_t_tilde=%f \nsigma_y_tilde=%f \nt_tilde=%f \nrho=%f \nlikelihood=%f \n",
  	   x_0_tilde,
  	   y_0_tilde,
  	   x_t_tilde,
  	   y_t_tilde,
  	   sigma_y_tilde,
  	   t_tilde,
  	   rho,
  	   likelihood);
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

    // likelihood
    std::getline(is, name, '=');
    std::getline(is, value, ';');
    point.likelihood = std::stod(value);

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

double likelihood_point_distance(const likelihood_point& lp1,
				 const likelihood_point& lp2,
				 const parameters_nominal& params);


double covariance(const likelihood_point& lp1,
		  const likelihood_point& lp2,
		  const parameters_nominal& params);

gsl_matrix* covariance_matrix(const std::vector<likelihood_point>& lps,
			      const parameters_nominal& params);

struct GaussianInterpolator {
  std::vector<likelihood_point> points_for_integration;
  std::vector<likelihood_point> points_for_interpolation;
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
  GaussianInterpolator(const GaussianInterpolator& rhs);
  
  GaussianInterpolator& operator=(const GaussianInterpolator& rhs);

  ~GaussianInterpolator() {
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
  inline void add_point_for_integration(const likelihood_point& new_point) {
    points_for_integration.push_back(new_point);
  }
  inline void add_point_for_interpolation(const likelihood_point& new_point) {
    points_for_interpolation.push_back(new_point);
    set_linalg_members();
  }

  gsl_matrix* covariance_matrix(const std::vector<likelihood_point>& lps,
				const parameters_nominal& params) const;

  double covariance(const likelihood_point& lp1,
		    const likelihood_point& lp2,
		    const parameters_nominal& params) const;

  double likelihood_point_distance(const likelihood_point& lp1,
				   const likelihood_point& lp2,
				   const parameters_nominal& params) const;

  double operator()(const likelihood_point& x);
  double prediction_variance(const likelihood_point& x);

};
