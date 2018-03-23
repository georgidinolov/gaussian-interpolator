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
  if (argc < 5 || argc > 5) {
    printf("You must provide input\n");
    printf("The input is: \nnumber points of data; \nnumber points per datum; \nfile name for interpolator; \nfile name for raw data \n");
    printf("It is wise to include the parameter values in the file name. We are using a fixed seed for random number generation.\n");
    exit(0);
  }
  int number_points_of_data = std::stod(argv[1]);
  int number_points_per_datum = std::stod(argv[2]);
  std::string input_file_name = argv[3];
  std::string raw_data_file_name = argv[4];

  omp_set_dynamic(0);
  omp_set_num_threads(3);
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
      if (points_for_kriging_not_flipped[i].size() > 6) {
	GPs_not_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
							     points_for_kriging_not_flipped[i]);
      } else {
	GPs_not_flipped[i] = GaussianInterpolatorWithChecker(points_for_integration,
							     points_for_kriging_flipped[i]);
      }
      GPs_not_flipped[i].optimize_parameters();

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

  
  
  return 0;
}
