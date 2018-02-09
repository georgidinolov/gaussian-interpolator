#include "GaussianInterpolator.hpp"
#include "src/multivariate-normal/MultivariateNormal.hpp"



likelihood_point_transformed::likelihood_point_transformed(const likelihood_point& lp)
{
  set_likelihood_point_transformed(lp);
}

likelihood_point_transformed::likelihood_point_transformed()
{
  likelihood_point lp = likelihood_point();
  set_likelihood_point_transformed(lp);
}

void likelihood_point_transformed::set_likelihood_point_transformed(const likelihood_point& lp)
{
  x_0_tilde_transformed = logit(lp.x_0_tilde);
  x_t_tilde_transformed = logit(lp.x_t_tilde);
  // 
  y_0_tilde_transformed = logit(lp.y_0_tilde);
  y_t_tilde_transformed = logit(lp.y_t_tilde);
  //
  sigma_y_tilde_transformed = log(lp.sigma_y_tilde);
  t_tilde_transformed = log(lp.t_tilde);
  rho_transformed = logit( (lp.rho + 1.0)/2.0 );
  //
  log_likelihood_transformed = lp.log_likelihood + 
    (log(lp.x_0_tilde) + log(1.0-lp.x_0_tilde)) +
    (log(lp.y_0_tilde) + log(1.0-lp.y_0_tilde)) +
    (log(lp.x_t_tilde) + log(1.0-lp.x_t_tilde)) +
    (log(lp.y_t_tilde) + log(1.0-lp.y_t_tilde)) +
    //
    log(lp.sigma_y_tilde) +
    log(lp.t_tilde) +
    //
    (log(2.0) + log(lp.rho) + log(1.0-lp.rho));
}

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
    gsl_vector_set(y, i, points_for_interpolation[i].log_likelihood);
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
    gsl_vector_set(y, i, points_for_interpolation[i].log_likelihood);
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
    gsl_vector_set(y, i, points_for_interpolation[i].log_likelihood);
    gsl_vector_set(mean, i, parameters.phi);
    gsl_vector_set(difference, i,
		   points_for_interpolation[i].log_likelihood-parameters.phi);
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

double GaussianInterpolator::optimization_wrapper(const std::vector<double> &x,
						  std::vector<double> &grad,
						  void * data)
{
  GaussianInterpolator * GP_ptr=
    reinterpret_cast<GaussianInterpolator*>(data);


  parameters_nominal params = parameters_nominal(x);

  // double sigma_2 = x[0];
  // double phi = x[1];
  // int nu = x[2];
  // double tau_2 = x[3];
  // std::vector<double> lower_triag_mat_as_vec = 
  //   std::vector<double> {x[4], //0
  // 			 0.0, x[5], //2
  // 			 0.0, 0.0, x[6], //5
  // 			 0.0, 0.0, 0.0, x[7], //9
  // 			 0.0, 0.0, 0.0, 0.0, x[8], //14
  // 			 0.0, 0.0, 0.0, 0.0, 0.0, x[9], //20
  // 			 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, x[10]}; //27
  
  // parameters_nominal params = parameters_nominal();
  // params.sigma_2 = sigma_2;
  // params.phi = phi;
  // params.nu = nu;
  // params.tau_2 = tau_2;
  // params.lower_triag_mat_as_vec = lower_triag_mat_as_vec;

  std::cout << params;
  GP_ptr->set_parameters(params);

  unsigned N = (GP_ptr->points_for_interpolation).size();
  gsl_vector* y = GP_ptr->y;
  gsl_vector* mu = GP_ptr->mean;
  gsl_matrix* C = GP_ptr->C;

  // std::cout << "y = ";
  // for (unsigned i=0; i<y->size; ++i) {
  //   std::cout << "(" << i << "," << gsl_vector_get(y,i) << ") ";
  //   // if (std::abs(std::isinf(gsl_vector_get(y,i)))) {
  //   //   gsl_vector_set(y,i,-1000.0);
  //   // }
  // }
  // std::cout << std::endl;
  
  MultivariateNormal mvtnorm = MultivariateNormal();
  double out = mvtnorm.log_dmvnorm(N,y,mu,C);
  // if (std::isnan(log(out))) {
  //   out = std::numeric_limits<double>::epsilon();
  // }

  std::cout << "ll = " << out << std::endl;
  return out;
}

void GaussianInterpolator::optimize_parameters()
{
  nlopt::opt opt(nlopt::LN_NELDERMEAD, 32);
  std::vector<double> lb = {0.00001, -HUGE_VAL, 5, 0.0001,
  			    0.00001, // diag element 1 min
  			    -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001,
			    -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001,
  			    -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001,
  			     -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001};
  
  std::vector<double> ub = {HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, // diag element 1 min
  			    HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
			    HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			     HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL};

  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_ftol_rel(0.0001);
  double maxf;

  opt.set_max_objective(GaussianInterpolator::optimization_wrapper,
  			this);

  std::vector<double> optimal_params = parameters.as_vector();
  // std::vector<double> optimal_params = std::vector<double> (11);
  // optimal_params[0] = parameters.sigma_2;
  // optimal_params[1] = parameters.phi;
  // optimal_params[2] = parameters.nu;
  // optimal_params[3] = parameters.tau_2;
  // unsigned counter = 0;
  // for (unsigned i=0; i<7; ++i) {
  //   for (unsigned j=0; j<=i; ++j) {
  //     if (i==j) {
  // 	optimal_params[3+1+i] = parameters.lower_triag_mat_as_vec[counter];
  //     }
  //     counter = counter + 1;
  //   }
  // }

  opt.optimize(optimal_params, maxf);
  parameters_nominal new_parameters = parameters_nominal(optimal_params);
  parameters = new_parameters;
  // parameters.sigma_2 = optimal_params[0];
  // parameters.phi = optimal_params[1];
  // parameters.nu = optimal_params[2];
  // parameters.tau_2 = optimal_params[3];

  // counter = 0;
  // for (unsigned i=0; i<7; ++i) {
  //   for (unsigned j=0; j<=i; ++j) {
  //     if (i==j) {
  // 	parameters.lower_triag_mat_as_vec[counter] = optimal_params[4+i];
  //     }
  //   }
  // }

  set_linalg_members();
}

double optimization_wrapper(const std::vector<double> &x,
			    std::vector<double> &grad,
			    void * data)
{
  GaussianInterpolator * GP_ptr=
    reinterpret_cast<GaussianInterpolator*>(data);
  
  parameters_nominal params = parameters_nominal(x);
  //   GP_ptr->parameters;
  // std::vector<double> params_as_vec = params.as_vector();
  // for (unsigned i=0; i<x.size(); ++i) {
  //   params_as_vec[i+1] = x[i];
  // }
  // params = parameters_nominal(params_as_vec);

  std::cout << params;
  GP_ptr->set_parameters(params);

  unsigned N = (GP_ptr->points_for_interpolation).size();
  gsl_vector* y = GP_ptr->y;
  gsl_vector* mu = GP_ptr->mean;
  gsl_matrix* C = GP_ptr->C;

  // std::cout << "y = ";
  // for (unsigned i=0; i<y->size; ++i) {
  //   std::cout << "(" << i << "," << gsl_vector_get(y,i) << ") ";
  //   // if (std::abs(std::isinf(gsl_vector_get(y,i)))) {
  //   //   gsl_vector_set(y,i,-1000.0);
  //   // }
  // }
  // std::cout << std::endl;
  
  MultivariateNormal mvtnorm = MultivariateNormal();
  double out = mvtnorm.log_dmvnorm(N,y,mu,C);
  // if (std::isnan(log(out))) {
  //   out = std::numeric_limits<double>::epsilon();
  // }

  std::cout << "ll = " << out << std::endl;
  return out;
}
