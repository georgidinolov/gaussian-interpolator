#include <algorithm>
#include <chrono>
#include <cmath>
#include "GaussianInterpolator.hpp"
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_rng.h>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdio.h>
#include <vector>

double log_likelihood(const std::vector<double> &x,
		      void * data) {
  // // //
  BrownianMotion * lp =
    reinterpret_cast<BrownianMotion*> (data);
  // // //

  double sigma_x = x[0];
  double sigma_y = x[1];
  double rho = x[2];

  double t = lp->get_t();
  // // //
  double out = -log(2*M_PI*1.0*sigma_x*sigma_y*std::sqrt(1-rho*rho)*t) +
    -1/(2*t*(1-rho*rho)) * (
  			    std::pow(lp->get_x_T() - lp->get_x_0(), 2)/std::pow(sigma_x,2.0) +
  			    std::pow(lp->get_y_T() - lp->get_y_0(), 2)/std::pow(sigma_y,2.0) -
  			    2*rho*(lp->get_x_T() - lp->get_x_0())*(lp->get_y_T() - lp->get_y_0())/
			    (sigma_x * sigma_y)
  			    ) ;

  return out;
}

double myfunc(const std::vector<double> &x,
	      std::vector<double> &grad,
	      void * data) {

  // if (!grad.empty()) {
  //   grad[0] = log_deriv_sigma(x, data);
  //   grad[1] = log_deriv_t(x, data);
  //   grad[2] = log_deriv_rho(x, data);
  // }

  std::vector<BrownianMotion> * lps =
    reinterpret_cast<std::vector<BrownianMotion>*> (data);

  double out = 0.0;
  for (unsigned i=0; i<lps->size(); ++i) {
    BrownianMotion lp = (*lps)[i];
    out = out + log_likelihood(x, &lp);
  }
  return out;
}

int main(int argc, char *argv[]) {
  if (argc < 5 || argc > 5) {
    printf("You must provide input\n");
    printf("The input is: \n\noutput file prefix; \nnumber points for interpolator; \nfile name for interpolator; \nfile name for parameters; \n");
    printf("It is wise to include the parameter values in the file name. We are using a fixed seed for random number generation.\n");
    exit(0);
  }

  std::string output_file_PREFIX = argv[1];
  int number_points_for_interpolator = std::stod(argv[2]);
  std::string input_file_name = argv[3];
  std::string parameters_file_name = argv[4];

  std::string output_file_name = output_file_PREFIX +  ".csv";

  std::cout << "output_file = " << output_file_name << std::endl;

  omp_set_dynamic(0);
  omp_set_num_threads(1);
  std::ifstream parameters_file(parameters_file_name);
  parameters_nominal params_for_GP_prior = parameters_nominal();
  parameters_file >> params_for_GP_prior;

  std::ifstream input_file(input_file_name);
  
  std::vector<likelihood_point> points_for_kriging_FLIPPED = std::vector<likelihood_point> (0);
  std::vector<likelihood_point> points_for_kriging_NOT_FLIPPED = std::vector<likelihood_point> (0);

  std::vector<likelihood_point> points_for_integration = std::vector<likelihood_point> (1);

  if (input_file.is_open()) {
    for (unsigned i=0; i<(number_points_for_interpolator); ++i) {
      likelihood_point current_lp = likelihood_point();
      input_file >> current_lp;
      if (i>=0) {
	if (current_lp.FLIPPED) { 
	  points_for_kriging_FLIPPED.push_back(current_lp);
	} else {
	  points_for_kriging_NOT_FLIPPED.push_back(current_lp);
	}
      }
    }
  }

  std::cout << "done with GP_prior\n";
  std::ifstream points_for_eval_file("likelihood_points_from_filter.csv");
  likelihood_point lp_for_eval = likelihood_point();
  GaussianInterpolator GP_prior = GaussianInterpolator(points_for_integration,
  						       points_for_kriging_FLIPPED,
  						       params_for_GP_prior);
  unsigned counter = 0;
  likelihood_point_transformed lp_for_eval_constant_tr = GP_prior.points_for_interpolation[counter];
  // while ((lp_for_eval_constant_tr.FLIPPED==false) & 
  // 	 (counter < GP_prior.points_for_interpolation.size())) {
  //   lp_for_eval_constant_tr = GP_prior.points_for_interpolation[counter];
  //   counter ++;
  // }
  likelihood_point lp_for_eval_constant = likelihood_point();
  lp_for_eval_constant = lp_for_eval_constant_tr;
  lp_for_eval = lp_for_eval_constant;

  std::cout << "lp_for_eval = \n";
  std::cout << lp_for_eval;

  // PLOTTING T //
  std::ofstream output_file(output_file_name);
  unsigned M = 100;
  double t_max = 5.0;
  std::vector<double> ts(M);
  std::generate(ts.begin(), ts.end(), [t=0.0, dt = t_max/M] () mutable {t = t + dt; return t; });

  lp_for_eval = lp_for_eval_constant;
  std::vector<double> lls_t = ts;
  std::vector<double> lls_t_std_dev = ts;
  for (unsigned i=0; i<lls_t.size(); ++i) {
    lp_for_eval.t_tilde = ts[i];
    lls_t[i] = GP_prior(lp_for_eval);
    lls_t_std_dev[i] = GP_prior.prediction_standard_dev(lp_for_eval);
  }
  std::cout << std::endl;

  output_file << "ts = c(";
  for (unsigned i=0; i<ts.size()-1; ++i) {
    output_file << ts[i] << ",";
  }
  output_file << ts[ts.size()-1] << ");\n";

  output_file << "lls.t = c(";
  for (unsigned i=0; i<lls_t.size()-1; ++i) {
    output_file << lls_t[i] << ",";
  }
  output_file << lls_t[lls_t.size()-1] << ");\n";

  output_file << "lls.t.std = c(";
  for (unsigned i=0; i<lls_t_std_dev.size()-1; ++i) {
    output_file << lls_t_std_dev[i] << ",";
  }
  output_file << lls_t_std_dev[lls_t_std_dev.size()-1] << ");\n";
  output_file << "par(mfcol=c(3,2), mar=c(4,4,4,4));\n";
  output_file << "plot(ts, lls.t, type=\"l\");\n";
  output_file << "lines(ts, lls.t + 2*(lls.t.std), col=2);\n";
  output_file << "lines(ts, lls.t - 2*(lls.t.std), col=2);\n";
  // PLOTTING T //

  // PLOTTING SIGMA //
  double sigma_max = 1.0;
  std::vector<double> sigma_tildes(M);
  std::generate(sigma_tildes.begin(), sigma_tildes.end(), [sigma=0.0, dsigma = sigma_max/M] () mutable {sigma = sigma + dsigma; return sigma; });

  lp_for_eval = lp_for_eval_constant;
  std::vector<double> lls_sigma = sigma_tildes;
  std::vector<double> lls_sigma_std_dev = sigma_tildes;
  for (unsigned i=0; i<lls_sigma.size(); ++i) {
    lp_for_eval.sigma_y_tilde = sigma_tildes[i];
    lls_sigma[i] = GP_prior(lp_for_eval);
    lls_sigma_std_dev[i] = GP_prior.prediction_standard_dev(lp_for_eval);
  }

  output_file << "sigmas = c(";
  for (unsigned i=0; i<sigma_tildes.size()-1; ++i) {
    output_file << sigma_tildes[i] << ",";
  }
  output_file << sigma_tildes[sigma_tildes.size()-1] << ");\n";

  output_file << "lls.sigma = c(";
  for (unsigned i=0; i<lls_sigma.size()-1; ++i) {
    output_file << lls_sigma[i] << ",";
  }
  output_file << lls_sigma[lls_sigma.size()-1] << ");\n";

  output_file << "lls.sigma.std = c(";
  for (unsigned i=0; i<lls_sigma_std_dev.size()-1; ++i) {
    output_file << lls_sigma_std_dev[i] << ",";
  }
  output_file << lls_sigma_std_dev[lls_sigma_std_dev.size()-1] << ");\n";

  output_file << "plot(sigmas, lls.sigma, type=\"l\");\n";
  output_file << "lines(sigmas, lls.sigma + 2*(lls.sigma.std), col=2);\n";
  output_file << "lines(sigmas, lls.sigma - 2*(lls.sigma.std), col=2);\n";
  // PLOTTING SIGMA //

  // PLOTTING RHO //
  double rho_max = 0.95;
  double rho_min = -rho_max;
  std::vector<double> rhos(M);
  std::generate(rhos.begin(), rhos.end(), [rho=-0.99, drho = (rho_max-rho_min)/M] () mutable {rho = rho + drho; return rho; });

  lp_for_eval = lp_for_eval_constant;
  std::vector<double> lls_rho = rhos;
  std::vector<double> lls_rho_std_dev = rhos;
  for (unsigned i=0; i<lls_rho.size(); ++i) {
    lp_for_eval.rho = rhos[i];
    lls_rho[i] = GP_prior(lp_for_eval);
    lls_rho_std_dev[i] = GP_prior.prediction_standard_dev(lp_for_eval);
  }

  output_file << "rhos = c(";
  for (unsigned i=0; i<rhos.size()-1; ++i) {
    output_file << rhos[i] << ",";
  }
  output_file << rhos[rhos.size()-1] << ");\n";

  output_file << "lls.rho = c(";
  for (unsigned i=0; i<lls_rho.size()-1; ++i) {
    output_file << lls_rho[i] << ",";
  }
  output_file << lls_rho[lls_rho.size()-1] << ");\n";

  output_file << "lls.rho.std = c(";
  for (unsigned i=0; i<lls_rho_std_dev.size()-1; ++i) {
    output_file << lls_rho_std_dev[i] << ",";
  }
  output_file << lls_rho_std_dev[lls_rho_std_dev.size()-1] << ");\n";

  output_file << "plot(rhos, lls.rho, type=\"l\");\n";
  output_file << "lines(rhos, lls.rho + 2*(lls.rho.std), col=2);\n";
  output_file << "lines(rhos, lls.rho - 2*(lls.rho.std), col=2);\n";
  // PLOTTING RHO //

  output_file << "plot(ts, exp(lls.t), type=\"l\");\n";
  output_file << "lines(ts, exp(lls.t) * exp(2*(lls.t.std)), col=2);\n";
  output_file << "lines(ts, exp(lls.t) / exp(2*(lls.t.std)), col=2);\n";

  output_file << "plot(sigmas, exp(lls.sigma), type=\"l\");\n";
  output_file << "lines(sigmas, exp(lls.sigma) * exp(2*(lls.sigma.std)), col=2);\n";
  output_file << "lines(sigmas, exp(lls.sigma) / exp(2*(lls.sigma.std)), col=2);\n";

  output_file << "plot(rhos, exp(lls.rho), type=\"l\");\n";
  output_file << "lines(rhos, exp(lls.rho) * exp(2*(lls.rho.std)), col=2);\n";
  output_file << "lines(rhos, exp(lls.rho) / exp(2*(lls.rho.std)), col=2);\n";

  return 0;
}
