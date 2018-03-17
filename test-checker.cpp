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


int main(int argc, char *argv[]) {
  if (argc < 4 || argc > 4) {
    printf("You must provide input\n");
    printf("The input is: \nnumber points for interpolator; \nfile name for interpolator; \nfile name for parameters; \n");
    printf("It is wise to include the parameter values in the file name. We are using a fixed seed for random number generation.\n");
    exit(0);
  }
  int number_points_for_interpolator = std::stod(argv[1]);
  std::string input_file_name = argv[2];
  std::string parameters_file_name = argv[3];

  omp_set_dynamic(0);
  omp_set_num_threads(1);
  std::ifstream parameters_file(parameters_file_name);
  parameters_nominal params_for_GP_prior = parameters_nominal();
  parameters_file >> params_for_GP_prior;

  std::ifstream input_file(input_file_name);
  
  std::vector<likelihood_point> points_for_kriging = std::vector<likelihood_point> (0);
  std::vector<likelihood_point> points_for_integration = std::vector<likelihood_point> (1);

  if (input_file.is_open()) {
    for (unsigned i=0; i<(number_points_for_interpolator); ++i) {
      likelihood_point current_lp = likelihood_point();
      input_file >> current_lp;
      if (i>=0) {
	points_for_kriging.push_back(current_lp);
      }
    }
  }

  std::cout << "done with GP_prior\n";
  std::ifstream points_for_eval_file("likelihood_points_from_filter.csv");
  likelihood_point lp_for_eval = likelihood_point();

  GaussianInterpolator* GP_prior = new GaussianInterpolator(points_for_integration,
						       points_for_kriging,
						       params_for_GP_prior);
  
  GaussianInterpolator GP_prior_w_checker;
  GP_prior_w_checker = *GP_prior;
  //   GaussianInterpolatorWithChecker(points_for_integration,
  // 				    points_for_kriging,
  // 				    params_for_GP_prior);

  // *GP_prior = GaussianInterpolatorWithChecker(points_for_integration,
  // 					     points_for_kriging,
  // 					     params_for_GP_prior);

  //  *GP_prior = *GP_prior_w_checker;

  // *GP_prior = GaussianInterpolator(points_for_integration,
  // 				    points_for_kriging,
  // 				   params_for_GP_prior);
  // GaussianInterpolator GP_prior_2(GP_prior_w_checker);

  return 0;
}
