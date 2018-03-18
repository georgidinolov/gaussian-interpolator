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
  
  std::vector<likelihood_point> points_for_kriging_flipped = std::vector<likelihood_point> (0);
  std::vector<likelihood_point> points_for_kriging_not_flipped = std::vector<likelihood_point> (0);
  std::vector<likelihood_point> points_for_integration = std::vector<likelihood_point> (1);

  if (input_file.is_open()) {
    for (unsigned i=0; i<(number_points_for_interpolator); ++i) {
      likelihood_point current_lp = likelihood_point();
      input_file >> current_lp;
      if (i>=0) {
	if (current_lp.FLIPPED) {
	  points_for_kriging_flipped.push_back(current_lp);
	} else {
	  points_for_kriging_not_flipped.push_back(current_lp);
	}
      }
    }
  }

  std::cout << "done with GP_prior\n";
  std::ifstream points_for_eval_file("likelihood_points_from_filter.csv");
  likelihood_point lp_for_eval = likelihood_point();

  GaussianInterpolatorWithChecker GP_prior_flipped(points_for_integration,
						   points_for_kriging_flipped,
						   params_for_GP_prior);

  GaussianInterpolatorWithChecker GP_prior_not_flipped(points_for_integration,
						       points_for_kriging_not_flipped,
						       params_for_GP_prior);

  for (const likelihood_point_transformed& current_lp_tr : GP_prior_not_flipped.points_for_interpolation) {
    std::cout << current_lp_tr.sigma_y_tilde_transformed << " ";
  }
  std::cout << std::endl;

  PointChecker point_checker_not_flipped(GP_prior_not_flipped.points_for_interpolation);
  PointChecker point_checker_flipped(GP_prior_flipped.points_for_interpolation);

  std::cout << point_checker_not_flipped;
  std::cout << point_checker_flipped;

  for (const likelihood_point_transformed& current_lp_tr : GP_prior_not_flipped.points_for_interpolation) {
    likelihood_point current_lp;
    current_lp = current_lp_tr;
    std::cout << "("
	      << point_checker_not_flipped.mahalanobis_distance(current_lp) << ","
	      << GP_prior_not_flipped.check_point(current_lp) << ") ";
  }
  std::cout << "\n" << std::endl;

  for (const likelihood_point_transformed& current_lp_tr : GP_prior_flipped.points_for_interpolation) {
    likelihood_point current_lp;
    current_lp = current_lp_tr;
    std::cout << "("
	      << point_checker_flipped.mahalanobis_distance(current_lp) << ","
	      << GP_prior_flipped.check_point(current_lp) << ") ";
  }
  std::cout << std::endl;
  
  return 0;
}
