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
    printf("The input is: \nnumber points of data; \nnumber points per datum; \nfile name for interpolator; \n");
    printf("It is wise to include the parameter values in the file name. We are using a fixed seed for random number generation.\n");
    exit(0);
  }
  int number_points_of_data = std::stod(argv[1]);
  int number_points_per_datum = std::stod(argv[2]);
  std::string input_file_name = argv[3];

  omp_set_dynamic(0);
  omp_set_num_threads(1);
  std::ifstream input_file(input_file_name);
  
  std::vector<std::vector<likelihood_point>> points_for_kriging_flipped = std::vector<std::vector<likelihood_point>> (number_points_of_data);
  std::vector<std::vector<likelihood_point>> points_for_kriging_not_flipped = std::vector<std::vector<likelihood_point>> (number_points_of_data);
  std::vector<likelihood_point> points_for_integration = std::vector<likelihood_point> (1);

  if (input_file.is_open()) {
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
  }

  std::cout << "done with GP_prior\n";
  std::vector<GaussianInterpolatorWithChecker> GPs_not_flipped(0);
  std::vector<GaussianInterpolatorWithChecker> GPs_flipped(0);
  for (unsigned i=0; i<number_points_of_data; ++i) {
    if (points_for_kriging_not_flipped[i].size() > 6) {
      GPs_not_flipped.push_back(GaussianInterpolatorWithChecker(points_for_integration,
      								points_for_kriging_not_flipped[i]));
    } else {
      GPs_not_flipped.push_back(GaussianInterpolatorWithChecker(points_for_integration,
								points_for_kriging_flipped[i]));
    }
    GPs_not_flipped[i].optimize_parameters();
    std::cout << GPs_not_flipped[i].parameters;

    if (points_for_kriging_flipped[i].size() > 6) {
      GPs_flipped.push_back(GaussianInterpolatorWithChecker(points_for_integration,
    								points_for_kriging_flipped[i]));
    } else {
      GPs_flipped.push_back(GaussianInterpolatorWithChecker(points_for_integration,
    								points_for_kriging_not_flipped[i]));
    }
    GPs_flipped[i].optimize_parameters();
    std::cout << GPs_flipped[i].parameters;
  }


  
  // GaussianInterpolatorWithChecker GP_prior_not_flipped(points_for_integration,
  // 						       points_for_kriging_not_flipped);
  // GP_prior_not_flipped.optimize_parameters();

  // for (const likelihood_point_transformed& current_lp_tr : GP_prior_not_flipped.points_for_interpolation) {
  //   std::cout << current_lp_tr.sigma_y_tilde_transformed << " ";
  // }
  // std::cout << std::endl;

  // PointChecker point_checker_not_flipped(GP_prior_not_flipped.points_for_interpolation);
  // PointChecker point_checker_flipped(GP_prior_flipped.points_for_interpolation);

  // std::cout << point_checker_not_flipped;
  // std::cout << point_checker_flipped;

  // for (const likelihood_point_transformed& current_lp_tr : GP_prior_not_flipped.points_for_interpolation) {
  //   likelihood_point current_lp;
  //   current_lp = current_lp_tr;
  //   std::cout << "("
  // 	      << point_checker_not_flipped.mahalanobis_distance(current_lp) << ","
  // 	      << GP_prior_not_flipped.check_point(current_lp) << ") ";
  // }
  // std::cout << "\n" << std::endl;

  // for (const likelihood_point_transformed& current_lp_tr : GP_prior_flipped.points_for_interpolation) {
  //   likelihood_point current_lp;
  //   current_lp = current_lp_tr;
  //   std::cout << "("
  // 	      << point_checker_flipped.mahalanobis_distance(current_lp) << ","
  // 	      << GP_prior_flipped.check_point(current_lp) << ") ";
  // }
  // std::cout << std::endl;
  
  return 0;
}
