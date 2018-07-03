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
  if (argc < 6 || argc > 6) {
    printf("You must provide input\n");
    printf("The input is: \nnumber of points of data; \nnumber points per datum; \nfile name for interpolator; \nfile name for raw data; \noutput files prefixes \n");
    printf("It is wise to include the parameter values in the file name. We are using a fixed seed for random number generation.\n");
    exit(0);
  }

  int number_points_of_data = std::stod(argv[1]);
  int number_points_per_datum = std::stod(argv[2]);
  std::string input_file_name = argv[3];
  std::string raw_data_file_name = argv[4];
  std::string output_file_prefix = argv[5];

  omp_set_dynamic(0);
  omp_set_num_threads(1);
  std::ifstream input_file(input_file_name);

  std::vector<std::vector<likelihood_point>> points_for_kriging_flipped = std::vector<std::vector<likelihood_point>> (number_points_of_data);
  std::vector<std::vector<likelihood_point>> points_for_kriging_not_flipped = std::vector<std::vector<likelihood_point>> (number_points_of_data);
  std::vector<likelihood_point> points_for_integration = std::vector<likelihood_point> (1);

  for (unsigned i=0; i<number_points_of_data; ++i) {
    for (unsigned m=0; m<number_points_per_datum; ++m) {
      likelihood_point current_lp = likelihood_point();
      input_file >> current_lp;

      if (current_lp.FLIPPED) {
	points_for_kriging_flipped[i].push_back(current_lp);
      } else {
	points_for_kriging_not_flipped[i].push_back(current_lp);
      }

    }

    std::cout << points_for_kriging_not_flipped[i].size() << "\n";
    std::cout << points_for_kriging_flipped[i].size() << "\n";
    std::cout << std::endl;
  }
  std::cout << "done with points for kriging\n";

  std::vector<GaussianInterpolatorWithChecker> GPs_not_flipped(number_points_of_data);
  std::vector<GaussianInterpolatorWithChecker> GPs_flipped(number_points_of_data);

  unsigned i;
#pragma omp parallel default(none) private(i) shared(points_for_integration, points_for_kriging_flipped, points_for_kriging_not_flipped, number_points_of_data, GPs_not_flipped, GPs_flipped)
  {
#pragma omp for
    for (i=0; i<number_points_of_data; ++i) {

      // std::string not_flipped_params_file_name =
      // 	"optimal-parameters-data-point-" + std::to_string(i) +
      // 	"-not-flipped-kriging-test-number-points-1040-rho_basis-0.90-sigma_x_basis-0.50-sigma_y_basis-0.15-dx_likelihood-0.001.csv";
      // std::ifstream not_flipped_params_file(not_flipped_params_file_name);
      // parameters_nominal not_flipped_params = parameters_nominal();
      // not_flipped_params_file >> not_flipped_params;

      if (points_for_kriging_not_flipped[i].size() > 6) {
	GPs_not_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
						  points_for_kriging_not_flipped[i]);
      } else {
	GPs_not_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
						  points_for_kriging_flipped[i]);
      }
      GPs_not_flipped[i].optimize_parameters();

      // std::string flipped_params_file_name =
      // 	"optimal-parameters-data-point-" + std::to_string(i) +
      // 	"-not-flipped-kriging-test-number-points-1040-rho_basis-0.90-sigma_x_basis-0.50-sigma_y_basis-0.15-dx_likelihood-0.001.csv";
      // std::ifstream flipped_params_file(flipped_params_file_name);
      // parameters_nominal flipped_params = parameters_nominal();
      // flipped_params_file >> flipped_params;

      if (points_for_kriging_flipped[i].size() > 6) {
	GPs_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
					      points_for_kriging_flipped[i]);
      } else {
	GPs_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
					      points_for_kriging_not_flipped[i]);
      }
      GPs_flipped[i].optimize_parameters();
    }
  }

  std::vector<std::string> fl_not_fl = std::vector<std::string> {"flipped", "not-flipped"};
  for (i=0; i<number_points_of_data; ++i) {
    for (unsigned j=0; j<2; ++j) {
      std::string output_file_name =
  	output_file_prefix + "-data-point-" + std::to_string(i) +
  	"-" + fl_not_fl[j] + ".csv";

      GaussianInterpolatorWithChecker GP_prior = GaussianInterpolatorWithChecker();
      if (j==0) {
      	GP_prior = GPs_flipped[i];
      } else {
      	GP_prior = GPs_not_flipped[i];
      }

      // PLOTTING T //
      std::ofstream output_file(output_file_name);
      unsigned M = 100;
      double t_max = 5.0;
      std::vector<double> ts(M);
      std::generate(ts.begin(), ts.end(), [t=0.0, dt = t_max/M] () mutable {t = t + dt; return t; });

      likelihood_point lp_for_eval_constant = likelihood_point();
      lp_for_eval_constant = GP_prior.points_for_interpolation[0];
      likelihood_point lp_for_eval = likelihood_point();

      lp_for_eval = lp_for_eval_constant;
      std::vector<double> lls_t = ts;
      std::vector<double> lls_t_std_dev = ts;
      for (unsigned i=0; i<lls_t.size(); ++i) {
      	lp_for_eval.t_tilde = ts[i];
      	lls_t[i] = GP_prior(lp_for_eval);
      	lls_t_std_dev[i] = GP_prior.prediction_standard_dev(lp_for_eval);
	std::cout << GP_prior.check_point(lp_for_eval) << " ";
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
    }
  }



<<<<<<< HEAD
=======
  
>>>>>>> 731fe37005be2203cfff7874d622c63367d21870
//   // PLOTTING T //
//   std::ofstream output_file(output_file_name);
//   unsigned M = 100;
//   double t_max = 5.0;
//   std::vector<double> ts(M);
//   std::generate(ts.begin(), ts.end(), [t=0.0, dt = t_max/M] () mutable {t = t + dt; return t; });

//   lp_for_eval = lp_for_eval_constant;
//   std::vector<double> lls_t = ts;
//   std::vector<double> lls_t_std_dev = ts;
//   for (unsigned i=0; i<lls_t.size(); ++i) {
//     lp_for_eval.t_tilde = ts[i];
//     lls_t[i] = GP_prior(lp_for_eval);
//     lls_t_std_dev[i] = GP_prior.prediction_standard_dev(lp_for_eval);
//   }
//   std::cout << std::endl;

//   output_file << "ts = c(";
//   for (unsigned i=0; i<ts.size()-1; ++i) {
//     output_file << ts[i] << ",";
//   }
//   output_file << ts[ts.size()-1] << ");\n";

//   output_file << "lls.t = c(";
//   for (unsigned i=0; i<lls_t.size()-1; ++i) {
//     output_file << lls_t[i] << ",";
//   }
//   output_file << lls_t[lls_t.size()-1] << ");\n";

//   output_file << "lls.t.std = c(";
//   for (unsigned i=0; i<lls_t_std_dev.size()-1; ++i) {
//     output_file << lls_t_std_dev[i] << ",";
//   }
//   output_file << lls_t_std_dev[lls_t_std_dev.size()-1] << ");\n";
//   output_file << "par(mfcol=c(3,2), mar=c(4,4,4,4));\n";
//   output_file << "plot(ts, lls.t, type=\"l\");\n";
//   output_file << "lines(ts, lls.t + 2*(lls.t.std), col=2);\n";
//   output_file << "lines(ts, lls.t - 2*(lls.t.std), col=2);\n";
//   // PLOTTING T //

//   // PLOTTING SIGMA //
//   double sigma_max = 1.0;
//   std::vector<double> sigma_tildes(M);
//   std::generate(sigma_tildes.begin(), sigma_tildes.end(), [sigma=0.0, dsigma = sigma_max/M] () mutable {sigma = sigma + dsigma; return sigma; });

//   lp_for_eval = lp_for_eval_constant;
//   std::vector<double> lls_sigma = sigma_tildes;
//   std::vector<double> lls_sigma_std_dev = sigma_tildes;
//   for (unsigned i=0; i<lls_sigma.size(); ++i) {
//     lp_for_eval.sigma_y_tilde = sigma_tildes[i];
//     lls_sigma[i] = GP_prior(lp_for_eval);
//     lls_sigma_std_dev[i] = GP_prior.prediction_standard_dev(lp_for_eval);
//   }

//   output_file << "sigmas = c(";
//   for (unsigned i=0; i<sigma_tildes.size()-1; ++i) {
//     output_file << sigma_tildes[i] << ",";
//   }
//   output_file << sigma_tildes[sigma_tildes.size()-1] << ");\n";

//   output_file << "lls.sigma = c(";
//   for (unsigned i=0; i<lls_sigma.size()-1; ++i) {
//     output_file << lls_sigma[i] << ",";
//   }
//   output_file << lls_sigma[lls_sigma.size()-1] << ");\n";

//   output_file << "lls.sigma.std = c(";
//   for (unsigned i=0; i<lls_sigma_std_dev.size()-1; ++i) {
//     output_file << lls_sigma_std_dev[i] << ",";
//   }
//   output_file << lls_sigma_std_dev[lls_sigma_std_dev.size()-1] << ");\n";

//   output_file << "plot(sigmas, lls.sigma, type=\"l\");\n";
//   output_file << "lines(sigmas, lls.sigma + 2*(lls.sigma.std), col=2);\n";
//   output_file << "lines(sigmas, lls.sigma - 2*(lls.sigma.std), col=2);\n";
//   // PLOTTING SIGMA //

//   // PLOTTING RHO //
//   double rho_max = 0.95;
//   double rho_min = -rho_max;
//   std::vector<double> rhos(M);
//   std::generate(rhos.begin(), rhos.end(), [rho=-0.99, drho = (rho_max-rho_min)/M] () mutable {rho = rho + drho; return rho; });

//   lp_for_eval = lp_for_eval_constant;
//   std::vector<double> lls_rho = rhos;
//   std::vector<double> lls_rho_std_dev = rhos;
//   for (unsigned i=0; i<lls_rho.size(); ++i) {
//     lp_for_eval.rho = rhos[i];
//     lls_rho[i] = GP_prior(lp_for_eval);
//     lls_rho_std_dev[i] = GP_prior.prediction_standard_dev(lp_for_eval);
//   }

//   output_file << "rhos = c(";
//   for (unsigned i=0; i<rhos.size()-1; ++i) {
//     output_file << rhos[i] << ",";
//   }
//   output_file << rhos[rhos.size()-1] << ");\n";

//   output_file << "lls.rho = c(";
//   for (unsigned i=0; i<lls_rho.size()-1; ++i) {
//     output_file << lls_rho[i] << ",";
//   }
//   output_file << lls_rho[lls_rho.size()-1] << ");\n";

//   output_file << "lls.rho.std = c(";
//   for (unsigned i=0; i<lls_rho_std_dev.size()-1; ++i) {
//     output_file << lls_rho_std_dev[i] << ",";
//   }
//   output_file << lls_rho_std_dev[lls_rho_std_dev.size()-1] << ");\n";

//   output_file << "plot(rhos, lls.rho, type=\"l\");\n";
//   output_file << "lines(rhos, lls.rho + 2*(lls.rho.std), col=2);\n";
//   output_file << "lines(rhos, lls.rho - 2*(lls.rho.std), col=2);\n";
//   // PLOTTING RHO //

//   output_file << "plot(ts, exp(lls.t), type=\"l\");\n";
//   output_file << "lines(ts, exp(lls.t) * exp(2*(lls.t.std)), col=2);\n";
//   output_file << "lines(ts, exp(lls.t) / exp(2*(lls.t.std)), col=2);\n";

//   output_file << "plot(sigmas, exp(lls.sigma), type=\"l\");\n";
//   output_file << "lines(sigmas, exp(lls.sigma) * exp(2*(lls.sigma.std)), col=2);\n";
//   output_file << "lines(sigmas, exp(lls.sigma) / exp(2*(lls.sigma.std)), col=2);\n";

//   output_file << "plot(rhos, exp(lls.rho), type=\"l\");\n";
//   output_file << "lines(rhos, exp(lls.rho) * exp(2*(lls.rho.std)), col=2);\n";
//   output_file << "lines(rhos, exp(lls.rho) / exp(2*(lls.rho.std)), col=2);\n";

  return 0;
}
