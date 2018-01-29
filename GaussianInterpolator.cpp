#include "GaussianInterpolator.hpp"

double GaussianInterpolator::likelihood_point_distance(const likelihood_point& lp1,
						       const likelihood_point& lp2,
						       const parameters_nominal& params) const
{
  gsl_vector* lp1_vec = lp1.as_gsl_vector();
  gsl_vector* lp2_vec = lp2.as_gsl_vector();
  gsl_vector_sub(lp1_vec, lp2_vec);

  gsl_matrix* L = params.lower_triag_matrix();
  gsl_blas_dtrmv(CblasLower, CblasTrans, CblasNonUnit, L, lp1_vec);
    
  gsl_vector* squared_scaled_diff = gsl_vector_alloc(7);
  gsl_vector_memcpy(squared_scaled_diff, lp1_vec);
  gsl_vector_mul(squared_scaled_diff, lp1_vec);

  double out = 0;
  for (unsigned i=0; i<7; ++i) {
    out = out +
      gsl_vector_get(squared_scaled_diff, i);
  }

  out = std::sqrt(out);
  
  gsl_vector_free(lp1_vec);
  gsl_vector_free(lp2_vec);
  gsl_matrix_free(L);
  gsl_vector_free(squared_scaled_diff);
  return out;
};

double GaussianInterpolator::covariance(const likelihood_point& lp1, // 
					const likelihood_point& lp2,
					const parameters_nominal& params) const
{

  gsl_error_handler_t* old_handler = gsl_set_error_handler_off();
  
  gsl_sf_result bessel_out;
  double dist = likelihood_point_distance(lp1,lp2,params);
  int status = gsl_sf_bessel_Kn_e(params.nu, dist, &bessel_out);
  double out = 0.0;

  if (status == GSL_EDOM) {  // dist is probably too close to zero
    out = params.tau_2 + params.sigma_2;
  } else {
    out = (params.tau_2 * std::pow(dist, params.nu) /
	   (std::pow(2, params.nu-1) * gsl_sf_gamma(params.nu))) *
      bessel_out.val;
  }

  gsl_set_error_handler(old_handler);

  return out;
};

gsl_matrix* GaussianInterpolator::covariance_matrix(const std::vector<likelihood_point>& lps,
						    const parameters_nominal& params) const
{
  unsigned dimension = lps.size();
  gsl_matrix* out = gsl_matrix_alloc(dimension,dimension);

  for (unsigned i=0; i<dimension; ++i) {
    for (unsigned j=0; j<=i; ++j) {
      double cov = 0;
      if (i==j) {
	cov = params.sigma_2 + covariance(lps[i], lps[j], params);
	gsl_matrix_set(out, i,j, cov);
      } else {
	cov = covariance(lps[i], lps[j], params);
	gsl_matrix_set(out, i,j, cov);
	gsl_matrix_set(out, j,i, cov);
      }
    }
  }

  return out;
};

GaussianInterpolator::GaussianInterpolator()
  : points_for_integration(std::vector<likelihood_point> (1)),
    points_for_interpolation(std::vector<likelihood_point> (1)),
    parameters(parameters_nominal()),
    C(NULL),
    Cinv(NULL),
    y(NULL),
    mean(NULL),
    difference(NULL),
    c(NULL)
{
  C = covariance_matrix(points_for_interpolation,
			parameters);
  Cinv = gsl_matrix_alloc(points_for_interpolation.size(),
			  points_for_interpolation.size());
  gsl_matrix_memcpy(Cinv, C);
  gsl_linalg_cholesky_decomp(Cinv);
  gsl_linalg_cholesky_invert(Cinv);

  y = gsl_vector_alloc(points_for_interpolation.size());
  mean = gsl_vector_alloc(points_for_interpolation.size());
  difference = gsl_vector_alloc(points_for_interpolation.size());
  c = gsl_vector_alloc(points_for_interpolation.size());

  for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
    gsl_vector_set(y, i, points_for_interpolation[i].likelihood);
    gsl_vector_set(mean, i, parameters.phi);
  }

  gsl_vector_memcpy(difference, y);
  gsl_vector_sub(difference, mean);
}

GaussianInterpolator::GaussianInterpolator(const std::vector<likelihood_point>& points_for_integration_in,
					   const std::vector<likelihood_point>& points_for_interpolation_in,
					   const parameters_nominal& params)
  : points_for_integration(points_for_integration_in),
    points_for_interpolation(points_for_interpolation_in),
    parameters(params),
    C(NULL),
    Cinv(NULL),
    y(NULL),
    mean(NULL),
    difference(NULL),
    c(NULL)
{
  C = covariance_matrix(points_for_interpolation,
			parameters);
  Cinv = gsl_matrix_alloc(points_for_interpolation.size(),
			  points_for_interpolation.size());
  gsl_matrix_memcpy(Cinv, C);
  gsl_linalg_cholesky_decomp(Cinv);
  gsl_linalg_cholesky_invert(Cinv);

  y = gsl_vector_alloc(points_for_interpolation.size());
  mean = gsl_vector_alloc(points_for_interpolation.size());
  c = gsl_vector_alloc(points_for_interpolation.size());
  difference = gsl_vector_alloc(points_for_interpolation.size());
    
  for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
    gsl_vector_set(y, i, points_for_interpolation[i].likelihood);
    gsl_vector_set(mean, i, parameters.phi);
  }

  gsl_vector_memcpy(difference, y);
  gsl_vector_sub(difference, mean);
}

GaussianInterpolator::GaussianInterpolator(const GaussianInterpolator& rhs)
{
  points_for_integration = rhs.points_for_integration;
  points_for_interpolation = rhs.points_for_interpolation;
  parameters = parameters;
  set_linalg_members();
}

GaussianInterpolator& GaussianInterpolator::operator=(const GaussianInterpolator& rhs)
{
    if (this==&rhs) {
      return *this;
    } else {
      points_for_integration = rhs.points_for_integration;
      points_for_interpolation = rhs.points_for_interpolation;
      parameters = parameters;
      set_linalg_members();

      return *this;
    }
}

void GaussianInterpolator::set_linalg_members()
{
  gsl_matrix_free(C);
  gsl_matrix_free(Cinv);
  gsl_vector_free(y);
  gsl_vector_free(mean);
  gsl_vector_free(difference);
  gsl_vector_free(c);

  C = covariance_matrix(points_for_interpolation,
			parameters);
  Cinv = gsl_matrix_alloc(points_for_interpolation.size(),
			  points_for_interpolation.size());

  // for (unsigned i=0; i<C->size1; ++i) {
  //   for (unsigned j=0; j<C->size2; ++j) {
  // 	std::cout << gsl_matrix_get(C, i,j) << " ";
  //   }
  //   std::cout << std::endl;
  // }
    
  gsl_matrix_memcpy(Cinv, C);
  gsl_linalg_cholesky_decomp(Cinv);
  gsl_linalg_cholesky_invert(Cinv);

  y = gsl_vector_alloc(points_for_interpolation.size());
  mean = gsl_vector_alloc(points_for_interpolation.size());
  c = gsl_vector_alloc(points_for_interpolation.size());
  difference = gsl_vector_alloc(points_for_interpolation.size());

  for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
    gsl_vector_set(y, i, points_for_interpolation[i].likelihood);
    gsl_vector_set(mean, i, parameters.phi);
    gsl_vector_set(difference, i,
		   points_for_interpolation[i].likelihood-parameters.phi);
  }
}

double GaussianInterpolator::operator()(const likelihood_point& x)
{
  double out = 0;
  double work [points_for_interpolation.size()];
  gsl_vector_view work_view = gsl_vector_view_array(work,
						    points_for_interpolation.size());
    
    
  for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
    gsl_vector_set(c, i, covariance(points_for_interpolation[i],
				    x,
				    parameters));
  }

  // y = alpha*op(A)*x + beta*y
  gsl_blas_dgemv(CblasTrans, // op(.) = ^T
		 1.0, // alpha = 1
		 Cinv, // A
		 c, // x
		 0.0, // beta = 0
		 &work_view.vector); // this is the output
    
  gsl_blas_ddot(&work_view.vector, difference, &out);

  out = out + parameters.phi;

  return out;
}

double GaussianInterpolator::prediction_variance(const likelihood_point& x)
{
  double out = 0;
  double work [points_for_interpolation.size()];
  gsl_vector_view work_view = gsl_vector_view_array(work,
						    points_for_interpolation.size());
    
    
  for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
    gsl_vector_set(c, i, covariance(points_for_interpolation[i],
				    x,
				    parameters));
  }

  // first compute Cinv * c
  // y = alpha*op(A)*x + beta*y = Cinv * c
  gsl_blas_dgemv(CblasNoTrans, // op(.) = I
		 1.0, // alpha = 1
		 Cinv, // A
		 c, // x
		 0.0, // beta = 0
		 &work_view.vector); // this is the output

    
  gsl_blas_ddot(&work_view.vector, c, &out);

  out = parameters.tau_2 - out;
  return out;
}
