#include <rotatingMHD/time_discretization.h>
#include <deal.II/base/exceptions.h>
namespace Step35
{
  using namespace dealii;
namespace TimeDiscretization
{
VSIMEXCoefficients::VSIMEXCoefficients(const unsigned int &order)
  : alpha(order+1),
    beta(order),
    gamma(order+1)
{
  Assert( ( ( order == 1) || ( order == 2 )),
    ExcMessage("Only VSIMEX of first and second order are currently implemented"));
}

VSIMEXCoefficients::VSIMEXCoefficients(const VSIMEXCoefficients &data)
  : alpha(data.alpha),
    beta(data.beta),
    gamma(data.gamma)
{}

VSIMEXCoefficients VSIMEXCoefficients::operator=(const VSIMEXCoefficients &data_to_copy)
{
  this->alpha = data_to_copy.alpha;
  this->beta  = data_to_copy.beta;
  this->gamma = data_to_copy.gamma;
  return *this;
}

void VSIMEXCoefficients::output()
{
  /* Hardcoded for VSIMEX of second order */
  std::cout << "Parameter set for a VSIMEX method of " 
            << beta.size()
            << "order" << std::endl;
  std::cout << "alpha = {" << alpha[0] << ", " << alpha[1] 
            << ", " << alpha[2] << "}" << std::endl;
  std::cout << "beta = {" << beta[0] << ", " << beta[1] << "}" << std::endl;
  std::cout << "gamma = {" << gamma[0] << ", " << gamma[1] 
            << ", " << gamma[2] << "}" << std::endl;             
}

VSIMEXMethod::VSIMEXMethod(const unsigned int         order,
                           const std::vector<double>  parameters,
                           const double               start_time,
                           const double               end_time,
                           const double               desired_start_step_size)
  : DiscreteTime(start_time, end_time, desired_start_step_size),
    order(order),
    parameters(parameters),
    coefficients(order),
    omega(1.0),
    k_n(desired_start_step_size),
    k_n_minus_1(desired_start_step_size)
{
  Assert( ( ( order == 1) || ( order == 2 )),
    ExcMessage("Only VSIMEX of first and second order are currently implemented"));
}

unsigned int VSIMEXMethod::get_order()
{
  return order;
}

std::vector<double> VSIMEXMethod::get_parameters()
{
  return parameters;
}

void VSIMEXMethod::get_coefficients(VSIMEXCoefficients &output)
{
  output = coefficients;
}

void VSIMEXMethod::update_coefficients()
{
  k_n             = get_next_step_size();
  k_n_minus_1     = get_previous_step_size();
  omega           = k_n / k_n_minus_1;

  switch (order)
  {
    case 1 :
      {
      static double gamma   = parameters[0];
      coefficients.alpha[0] = 1.0 / k_n;
      coefficients.alpha[1] = - 1.0 / k_n;
      coefficients.beta[0]  = 1.0;
      coefficients.gamma[0] = ( 1.0 - gamma);
      coefficients.gamma[1] = gamma;
      break;
      }
    case 2 :
      {
      static double gamma   = parameters[0];
      static double c       = parameters[1];
      coefficients.alpha[0] = (2.0 * gamma - 1.0) * omega * omega / 
                                (1.0 + omega) / k_n;
      coefficients.alpha[1] = ((1.0 - 2.0 * gamma) * omega - 1.0) / k_n;
      coefficients.alpha[2] = (1.0 + 2.0 * gamma * omega) / 
                                (1.0 + omega) / k_n;
      coefficients.beta[0]  = - gamma * omega;
      coefficients.beta[1]  = 1.0 + gamma * omega;
      coefficients.gamma[0] = c / 2.0;
      coefficients.gamma[1] = 1.0 - gamma - (1.0 + 1.0 / omega) * c / 2.0; 
      coefficients.gamma[2] = gamma + c / (2.0 * omega); 
      break;
      }
    default :
     Assert(false,
      ExcMessage("Only VSIMEX of first and second order are currently implemented"));
  }
  coefficients.output();
}

} // namespace TimeDiscretiation
} // namespace Step35
