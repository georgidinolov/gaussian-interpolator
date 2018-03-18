#include "GaussianInterpolator.hpp"
#include "src/multivariate-normal/MultivariateNormal.hpp"

double gp_logit(double p) {
  return log(p/(1.0-p));
}

double gp_logit_inv(double r) {
  return exp(r)/(1 + exp(r));
}

likelihood_point& likelihood_point::operator=(const likelihood_point_transformed& rhs)
{
  x_0_tilde = gp_logit_inv(rhs.x_0_tilde_transformed);
  y_0_tilde = gp_logit_inv(rhs.y_0_tilde_transformed);
  //
  x_t_tilde = gp_logit_inv(rhs.x_t_tilde_transformed);
  y_t_tilde = gp_logit_inv(rhs.y_t_tilde_transformed);
  //
  sigma_y_tilde = exp(rhs.sigma_y_tilde_transformed);
  t_tilde = exp(rhs.t_tilde_transformed);
  rho = 2.0*gp_logit_inv(rhs.rho_transformed) - 1.0;
  //
  log_likelihood = rhs.log_likelihood_transformed +
    -log(x_0_tilde) - log(1-x_0_tilde) +
    -log(y_0_tilde) - log(1-y_0_tilde) +
    -log(x_t_tilde) - log(1-x_t_tilde) +
    -log(y_t_tilde) - log(1-y_t_tilde) +
    -log(sigma_y_tilde) +
    -log(t_tilde) +
    +log(2.0) - log(1-std::pow(rho,2));

  return *this;
}

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
  x_0_tilde_transformed = gp_logit(lp.x_0_tilde);
  x_t_tilde_transformed = gp_logit(lp.x_t_tilde);
  //
  y_0_tilde_transformed = gp_logit(lp.y_0_tilde);
  y_t_tilde_transformed = gp_logit(lp.y_t_tilde);
  //
  sigma_y_tilde_transformed = log(lp.sigma_y_tilde);
  t_tilde_transformed = log(lp.t_tilde);
  rho_transformed = gp_logit( (lp.rho + 1.0)/2.0 );
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
    log(lp.rho+1.0) + log((1.0-lp.rho)/2.0);
}

gsl_vector* likelihood_point_transformed::as_gsl_vector() const
  {
    gsl_vector* out = gsl_vector_alloc(7);
    gsl_vector_set(out, 0, x_0_tilde_transformed);
    gsl_vector_set(out, 1, y_0_tilde_transformed);
    //
    gsl_vector_set(out, 2, x_t_tilde_transformed);
    gsl_vector_set(out, 3, y_t_tilde_transformed);
    //
    gsl_vector_set(out, 4, sigma_y_tilde_transformed);
    gsl_vector_set(out, 5, t_tilde_transformed);
    gsl_vector_set(out, 6, rho_transformed);

    return out;
  }

std::ostream& operator<<(std::ostream& os,
			 const likelihood_point_transformed& point)
  {
    os << "x_0_tilde=" << std::scientific << std::setprecision(14)
       << point.x_0_tilde_transformed << ";\n";
    os << "y_0_tilde=" << std::scientific << std::setprecision(14)
       << point.y_0_tilde_transformed << ";\n";
    //
    os << "x_t_tilde=" << std::scientific << std::setprecision(14)
       << point.x_t_tilde_transformed << ";\n";
    os << "y_t_tilde=" << std::scientific << std::setprecision(14)
       << point.y_t_tilde_transformed << ";\n";
    //
    os << "sigma_y_tilde=" << std::scientific << std::setprecision(14)
       << point.sigma_y_tilde_transformed << ";\n";
    os << "t_tilde=" << std::scientific << std::setprecision(14)
       << point.t_tilde_transformed << ";\n";
    os << "rho=" << std::scientific << std::setprecision(14)
       << point.rho_transformed << ";\n";
    //
    os << "log_likelihood=" << std::scientific << std::setprecision(14)
       << point.log_likelihood_transformed << ";\n";
    os << "FLIPPED=" << std::scientific << std::setprecision(14)
       << point.FLIPPED << ";\n";

    return os;
  }

double GaussianInterpolator::likelihood_point_distance(const likelihood_point_transformed& lp1,
						       const likelihood_point_transformed& lp2,
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

double GaussianInterpolator::covariance(const likelihood_point_transformed& lp1, //
					const likelihood_point_transformed& lp2,
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

  // if (std::isnan(out)) {
  //   std::cout << "lp1 = " << lp1;
  //   std::cout << "lp2 = " << lp2;
  //   abort();
  // }

  return out;
};

gsl_matrix* GaussianInterpolator::covariance_matrix(const std::vector<likelihood_point_transformed>& lps,
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
  : points_for_integration(std::vector<likelihood_point_transformed> (1)),
    points_for_interpolation(std::vector<likelihood_point_transformed> (1)),
    parameters(parameters_nominal()),
    C(NULL),
    Cinv(NULL),
    y(NULL),
    mean(NULL),
    difference(NULL),
    c(NULL)
{
  printf("in GaussianInterpolator::default ctor\n");
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
    likelihood_point_transformed lpt = points_for_interpolation[i];
    gsl_vector_set(y, i, lpt.log_likelihood_transformed);
    gsl_vector_set(mean, i, parameters.phi);
  }

  gsl_vector_memcpy(difference, y);
  gsl_vector_sub(difference, mean);
}

GaussianInterpolator::
GaussianInterpolator(const std::vector<likelihood_point>& points_for_integration_in,
		     const std::vector<likelihood_point>& points_for_interpolation_in,
		     const parameters_nominal& params)
  : points_for_integration(),
    points_for_interpolation(),
    parameters(params),
    C(NULL),
    Cinv(NULL),
    y(NULL),
    mean(NULL),
    difference(NULL),
    c(NULL)
{
  printf("in GaussianInterpolator::ctor\n");
  points_for_integration = std::vector<likelihood_point_transformed> (0);
  //
  for (likelihood_point current_lp : points_for_integration_in) {
    points_for_integration.push_back(likelihood_point_transformed(current_lp));
  }

  points_for_interpolation = std::vector<likelihood_point_transformed> (0);
  for (likelihood_point current_lp : points_for_interpolation_in) {
    if ( !std::isnan(std::abs(current_lp.log_likelihood)) &&
	 !std::isinf(std::abs(current_lp.log_likelihood)) &&
	 (current_lp.log_likelihood < 3.0) &&
	 (current_lp.log_likelihood > -9.0) &&
 	 (current_lp.sigma_y_tilde >= 0.40) &&
 	 (current_lp.t_tilde >= 0.30) &&
 	 (std::abs(current_lp.rho) <= 0.95))
  	{
	  likelihood_point_transformed current_lp_tr(current_lp);
  	  points_for_interpolation.push_back(current_lp_tr);
  	}
  }

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
    likelihood_point_transformed lpt = points_for_interpolation[i];
    gsl_vector_set(y, i, lpt.log_likelihood_transformed);
    gsl_vector_set(mean, i, parameters.phi);
  }

  gsl_vector_memcpy(difference, y);
  gsl_vector_sub(difference, mean);
}

GaussianInterpolator::GaussianInterpolator(const GaussianInterpolator& rhs)
  : C(NULL),
    Cinv(NULL),
    y(NULL),
    mean(NULL),
    difference(NULL),
    c(NULL)
{
  points_for_integration = rhs.points_for_integration;
  points_for_interpolation = rhs.points_for_interpolation;
  parameters = parameters;

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
    likelihood_point_transformed lpt = points_for_interpolation[i];
    gsl_vector_set(y, i, lpt.log_likelihood_transformed);
    gsl_vector_set(mean, i, parameters.phi);
  }

  gsl_vector_memcpy(difference, y);
  gsl_vector_sub(difference, mean);
}

GaussianInterpolator& GaussianInterpolator::operator=(const GaussianInterpolator& rhs)
{
  printf("in GaussianInterpolator::operator=\n");
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
    gsl_vector_set(y, i, points_for_interpolation[i].log_likelihood_transformed);
    gsl_vector_set(mean, i, parameters.phi);
    gsl_vector_set(difference, i,
		   points_for_interpolation[i].log_likelihood_transformed-parameters.phi);
  }
}

double GaussianInterpolator::operator()(const likelihood_point& x)
{
  likelihood_point_transformed y(x);

  double out = 0;
  double work [points_for_interpolation.size()];
  gsl_vector_view work_view = gsl_vector_view_array(work,
						    points_for_interpolation.size());


  for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
    gsl_vector_set(c, i, covariance(points_for_interpolation[i],
				    y,
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

  out = out + parameters.phi +
    -log(x.x_0_tilde) - log(1-x.x_0_tilde) +
    -log(x.y_0_tilde) - log(1-x.y_0_tilde) +
    -log(x.x_t_tilde) - log(1-x.x_t_tilde) +
    -log(x.y_t_tilde) - log(1-x.y_t_tilde) +
    -log(x.sigma_y_tilde) +
    -log(x.t_tilde) +
    +log(2.0) - log(1-std::pow(x.rho,2));

  return out;
}

double GaussianInterpolator::prediction_standard_dev(const likelihood_point& x)
{
  likelihood_point_transformed y(x);
  double out = 0;
  double work [points_for_interpolation.size()];
  gsl_vector_view work_view = gsl_vector_view_array(work,
						    points_for_interpolation.size());


  for (unsigned i=0; i<points_for_interpolation.size(); ++i) {
    gsl_vector_set(c, i, covariance(points_for_interpolation[i],
				    y,
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

  out = std::sqrt(parameters.tau_2 - out);

  return out;
}

double GaussianInterpolator::optimization_wrapper(const std::vector<double> &x,
						  std::vector<double> &grad,
						  void * data)
{
  GaussianInterpolator * GP_ptr=
    reinterpret_cast<GaussianInterpolator*>(data);

  parameters_nominal params = //parameters_nominal(x);
    GP_ptr->parameters;

  params.phi = x[0];
  params.nu = x[1];
  params.tau_2 = x[2];
  unsigned counter_mat_as_vec=0;
  unsigned counter_param=0;
  std::vector<double> lower_triag_mat_as_vec = params.lower_triag_mat_as_vec;
  for (unsigned i=0; i<7; ++i) {
    for (unsigned j=0; j<=i; ++j) {
      if ((i>=4) && (j==i)) {
	lower_triag_mat_as_vec[counter_mat_as_vec] = x[3 + counter_param];
	counter_param = counter_param + 1;
      }
      // if ((i<=3) & (j==i)) {
      // 	lower_triag_mat_as_vec[counter_mat_as_vec] = x[3 + counter_param];
      // 	counter_param = counter_param + 1;
      // } else if ((i>3) & (j==i)) {
      // 	lower_triag_mat_as_vec[counter_mat_as_vec] = x[3 + counter_param];
      // 	counter_param = counter_param + 1;
      // }
      counter_mat_as_vec = counter_mat_as_vec + 1;
    }
  }
  params.lower_triag_mat_as_vec = lower_triag_mat_as_vec;

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
  nlopt::opt opt(nlopt::LN_NELDERMEAD, 6);
  std::vector<double> lb = {/*0.00001,*/ -HUGE_VAL, 5, 0.0001,
  			    /* 0.00001, // diag element 1 min*/
  			    /*-HUGE_VAL, 0.00001,*/
  			    /*-HUGE_VAL, -HUGE_VAL, 0.00001,*/
  			    /*-HUGE_VAL, -HUGE_VAL, -HUGE_VAL, 0.00001,*/
			    /*-HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL,*/ 0.00001,
  			    /*-HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL,*/ 0.00001,
			    /*-HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL, -HUGE_VAL,*/ 0.00001};

  std::vector<double> ub = {/*HUGE_VAL,*/ HUGE_VAL, HUGE_VAL, HUGE_VAL,
  			    /* HUGE_VAL,*/ // diag element 1 min
  			    /*HUGE_VAL, HUGE_VAL,*/
  			    /*HUGE_VAL, HUGE_VAL, HUGE_VAL,*/
  			    /*HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,*/
			    /*HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,*/ HUGE_VAL,
  			    /*HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,*/ HUGE_VAL,
			    /*HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL, HUGE_VAL,*/ HUGE_VAL};

  opt.set_lower_bounds(lb);
  opt.set_upper_bounds(ub);
  opt.set_ftol_rel(0.0001);
  double maxf;

  opt.set_max_objective(GaussianInterpolator::optimization_wrapper,
  			this);

  std::vector<double> optimal_params = std::vector<double> (6);
  // optimal_params[0] = parameters.sigma_2;
  optimal_params[0] = parameters.phi;
  optimal_params[1] = parameters.nu;
  optimal_params[2] = parameters.tau_2;
  unsigned counter_mat_as_vec = 0;
  unsigned counter_param = 0;
  for (unsigned i=0; i<7; ++i) {
    for (unsigned j=0; j<=i; ++j) {
      if ( (i>=4) && (j==i)) {
	optimal_params[3+counter_param] = parameters.lower_triag_mat_as_vec[counter_mat_as_vec];
	counter_param = counter_param + 1;
      }
      // if ((i<=3) & (i==j)) {
      // 	optimal_params[3+counter_param] = parameters.lower_triag_mat_as_vec[counter_mat_as_vec];
      // 	counter_param = counter_param + 1;
      // }  else if ((i>3) & (j==i)) {
      // 	optimal_params[3+counter_param] = parameters.lower_triag_mat_as_vec[counter_mat_as_vec];
      // 	counter_param = counter_param + 1;
      // }
      counter_mat_as_vec = counter_mat_as_vec + 1;
    }
  }


  opt.optimize(optimal_params, maxf);

  // parameters_nominal new_parameters = parameters_nominal(optimal_params);
  // parameters = new_parameters;

  parameters.phi = optimal_params[0];
  parameters.nu = optimal_params[1];
  parameters.tau_2 = optimal_params[2];

  counter_mat_as_vec = 0;
  counter_param = 0;
  for (unsigned i=0; i<7; ++i) {
    for (unsigned j=0; j<=i; ++j) {
      if ((i>=4) && (j==i)) {
	parameters.lower_triag_mat_as_vec[counter_mat_as_vec] = optimal_params[3+counter_param];
	counter_param = counter_param + 1;
      }
      // if ((i<=3) & (j==i)) {
      // 	parameters.lower_triag_mat_as_vec[counter_mat_as_vec] = optimal_params[3+counter_param];
      // 	counter_param = counter_param + 1;
      // } else if ((i>3) & (j==i)) {
      // 	parameters.lower_triag_mat_as_vec[counter_mat_as_vec] = optimal_params[3+counter_param];
      // 	counter_param = counter_param + 1;
      // }
      counter_mat_as_vec = counter_mat_as_vec + 1;
    }
  }
  set_linalg_members();
}

// CHECKER START
PointChecker::PointChecker()
  : mu_mean(NULL),
    sigma_cov(NULL),
    sigma_cov_inv(NULL)
{
  mu_mean = gsl_vector_calloc(3);
  sigma_cov = gsl_matrix_calloc(3,3);
  sigma_cov_inv = gsl_matrix_calloc(3,3);
}

PointChecker::PointChecker(const PointChecker& rhs)
  : mu_mean(NULL),
    sigma_cov(NULL),
    sigma_cov_inv(NULL)
{
  mu_mean = gsl_vector_alloc(rhs.mu_mean->size);
  sigma_cov = gsl_matrix_alloc(rhs.sigma_cov->size1,
			       rhs.sigma_cov->size2);
  sigma_cov_inv = gsl_matrix_alloc(rhs.sigma_cov_inv->size1,
				 rhs.sigma_cov_inv->size2);

  gsl_vector_memcpy(mu_mean, rhs.mu_mean);
  gsl_matrix_memcpy(sigma_cov, rhs.sigma_cov);
  gsl_matrix_memcpy(sigma_cov_inv, rhs.sigma_cov_inv);
}

PointChecker& PointChecker::operator=(const PointChecker& rhs)
{
  if (this==&rhs) {
    return *this;
  } else {
    gsl_vector_free(mu_mean);
    gsl_matrix_free(sigma_cov);
    gsl_matrix_free(sigma_cov_inv);

    mu_mean = gsl_vector_alloc(rhs.mu_mean->size);
    sigma_cov = gsl_matrix_alloc(rhs.sigma_cov->size1,
				 rhs.sigma_cov->size2);
    sigma_cov_inv = gsl_matrix_alloc(rhs.sigma_cov_inv->size1,
				     rhs.sigma_cov_inv->size2);

    gsl_vector_memcpy(mu_mean, rhs.mu_mean);
    gsl_matrix_memcpy(sigma_cov, rhs.sigma_cov);
    gsl_matrix_memcpy(sigma_cov_inv, rhs.sigma_cov_inv);

    return *this;
  }
}

PointChecker::PointChecker(const std::vector<likelihood_point_transformed>& lptrs)
  : mu_mean(NULL),
    sigma_cov(NULL),
    sigma_cov_inv(NULL)
{
  mu_mean = gsl_vector_calloc(3);
  sigma_cov = gsl_matrix_calloc(3,3);
  sigma_cov_inv = gsl_matrix_calloc(3,3);

  // calculating sufficient statistics
  gsl_vector* sums = gsl_vector_calloc(3);
  gsl_matrix* sums_of_squares = gsl_matrix_calloc(3,3);

  for (const likelihood_point_transformed& current_lp : lptrs) {
    gsl_vector* current_lp_gsl = current_lp.as_gsl_vector();

    for (unsigned i=0; i<3; ++i) {
      double current_sum_elem = gsl_vector_get(sums, i);
      gsl_vector_set(sums, i,
		     current_sum_elem + gsl_vector_get(current_lp_gsl, 4+i));

      for (unsigned j=0; j<3; ++j) {
	double current_sum_sq_elem = gsl_matrix_get(sums_of_squares,i,j);
	gsl_matrix_set(sums_of_squares, i, j,
		       current_sum_sq_elem +
		       gsl_vector_get(current_lp_gsl, 4+i)*
		       gsl_vector_get(current_lp_gsl, 4+j));

      }
    }

    gsl_vector_free(current_lp_gsl);
  }

  // calculating mean
  gsl_vector_memcpy(mu_mean, sums);
  gsl_vector_scale(mu_mean, 1.0/lptrs.size());

  // calculating cov
  for (unsigned i=0; i<3; ++i) {
    for (unsigned j=0; j<3; ++j) {
      gsl_matrix_set(sigma_cov, i,j,
		     gsl_matrix_get(sums_of_squares,i,j)/(1.0*lptrs.size()) -
		     gsl_vector_get(mu_mean,i)*gsl_vector_get(mu_mean,j));
    }
  }

  // calculating inv cov
  gsl_matrix_memcpy(sigma_cov_inv, sigma_cov);
  gsl_linalg_cholesky_decomp(sigma_cov_inv);
  gsl_linalg_cholesky_invert(sigma_cov_inv);

  gsl_matrix_free(sums_of_squares);
  gsl_vector_free(sums);
}

double PointChecker::mahalanobis_distance(const likelihood_point& lp) const
{
  likelihood_point_transformed lp_tr(lp);
  gsl_vector* lp_tr_gsl = lp_tr.as_gsl_vector();
  gsl_vector* diff2 = gsl_vector_calloc(3);
  gsl_vector_view lp_tr_gsl_subvec = gsl_vector_subvector(lp_tr_gsl,
							  4, // offset
							  3); // size
  gsl_vector_sub(&lp_tr_gsl_subvec.vector, mu_mean);
  gsl_vector_memcpy(diff2, &lp_tr_gsl_subvec.vector);
  //
  // y = alpha*A + beta*y, where A is symmetric
  gsl_blas_dsymv(CblasUpper, // upper
		 1.0, // alpha
		 sigma_cov_inv, // A
		 diff2, // x
		 0.0,
		 &lp_tr_gsl_subvec.vector);

  double ss = 0;
  gsl_blas_ddot(diff2, &lp_tr_gsl_subvec.vector, &ss);
  //
  gsl_vector_free(lp_tr_gsl);
  gsl_vector_free(diff2);

  return std::sqrt(ss);
}

std::ostream& operator<<(std::ostream& os,
			 const PointChecker& checker)
{
  os << "mu_mean = (";
  for (unsigned i=0; i<checker.mu_mean->size-1; ++i) {
    os << gsl_vector_get(checker.mu_mean, i) << " ";
  }
  os << gsl_vector_get(checker.mu_mean, checker.mu_mean->size-1) << ")\n";

  os << "sigma_cov = \n";
  for (unsigned i=0; i<checker.sigma_cov->size1; ++i) {
    for (unsigned j=0; j<checker.sigma_cov->size2; ++j) {
      os << gsl_matrix_get(checker.sigma_cov, i, j) << " ";
    }
    os << "\n";
  }
  os << "\n";

  return os;
}

// GAUSSIAN INTERPOLATOR WITH CHECKER START
GaussianInterpolatorWithChecker::GaussianInterpolatorWithChecker()
  : GaussianInterpolator(),
    point_checker(PointChecker())
{}

GaussianInterpolatorWithChecker::
GaussianInterpolatorWithChecker(const std::vector<likelihood_point>& points_for_integration_in,
				const std::vector<likelihood_point>& points_for_interpolation_in,
				const parameters_nominal& params)
  : GaussianInterpolator(points_for_integration_in,
			 points_for_interpolation_in,
			 params),
    point_checker(PointChecker(points_for_interpolation))
{
  printf("in GaussianInterpolatorWithChecker::ctor\n");
}

GaussianInterpolatorWithChecker::
GaussianInterpolatorWithChecker(const GaussianInterpolatorWithChecker& rhs)
  : GaussianInterpolator(rhs),
    point_checker(PointChecker(points_for_interpolation))
{}

// GaussianInterpolatorWithChecker::
// GaussianInterpolatorWithChecker(const GaussianInterpolator& rhs)
//   : GaussianInterpolator(rhs)
// {}

GaussianInterpolatorWithChecker& GaussianInterpolatorWithChecker::operator=(const GaussianInterpolatorWithChecker& rhs)
{
    if (this==&rhs) {
      return *this;
    } else {
      GaussianInterpolator::operator=(rhs);
      point_checker = PointChecker(rhs.points_for_interpolation);

      return *this;
    }
}

double GaussianInterpolatorWithChecker::check_point(const likelihood_point& lp) const
{
  return point_checker.mahalanobis_distance(lp);
}
