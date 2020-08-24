#include <rotatingMHD/time_discretization.h>
#include <deal.II/base/exceptions.h>
namespace RMHD
{
  using namespace dealii;
namespace TimeDiscretization
{
VSIMEXCoefficients::VSIMEXCoefficients()
{}

VSIMEXCoefficients::VSIMEXCoefficients(const unsigned int &order)
  : alpha(order+1),
    beta(order),
    gamma(order+1),
    phi(2)
{
  Assert( ( ( order == 1) || ( order == 2 )),
    ExcMessage("Only VSIMEX of first and second order are currently implemented"));
}

VSIMEXCoefficients::VSIMEXCoefficients(const VSIMEXCoefficients &data)
  : alpha(data.alpha),
    beta(data.beta),
    gamma(data.gamma),
    phi(data.phi)
{}

VSIMEXCoefficients VSIMEXCoefficients::operator=(const VSIMEXCoefficients &data_to_copy)
{
  this->alpha = data_to_copy.alpha;
  this->beta  = data_to_copy.beta;
  this->gamma = data_to_copy.gamma;
  this->phi = data_to_copy.phi;
  return *this;
}

void VSIMEXCoefficients::reinit(const unsigned int order)
{
  *this = VSIMEXCoefficients(order);
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

VSIMEXMethod::VSIMEXMethod(const VSIMEXScheme         scheme,
                           const double               start_time,
                           const double               end_time,
                           const double               desired_start_step_size,
                           const double               timestep_lower_bound,
                           const double               timestep_upper_bound)
  : DiscreteTime(start_time, end_time, desired_start_step_size),
    scheme(scheme),
    omega(1.0),
    timestep(desired_start_step_size),
    old_timestep(desired_start_step_size),
    timestep_lower_bound(timestep_lower_bound),
    timestep_upper_bound(timestep_upper_bound)
{
  Assert(((timestep > timestep_lower_bound) || 
          (timestep < timestep_upper_bound)),
      ExcMessage(
        "The desired start step is not inside the give bounded range."));
  switch (scheme)
  {
    case VSIMEXScheme::FE :
      order = 1;
      parameters.resize(order);
      parameters[0] = 0.;
      break;
    case VSIMEXScheme::CNFE :
      order = 1;
      parameters.resize(order);
      parameters[0] = 0.5;
      break;
    case VSIMEXScheme::BEFE :
      order = 1;
      parameters.resize(order);
      parameters[0] = 1.0;
      break;
    case VSIMEXScheme::BDF2 :
      order = 2;
      parameters.resize(order);
      parameters[0] = 1.0;
      parameters[1] = 0.0;
      break;
    case VSIMEXScheme::CNAB :
      order = 2;
      parameters.resize(order);
      parameters[0] = 0.5;
      parameters[1] = 0.0;
      break;
    case VSIMEXScheme::mCNAB :
      order = 2;
      parameters.resize(order);
      parameters[0] = 0.5;
      parameters[1] = 1.0/8.0;
      break;
    case VSIMEXScheme::CNLF :
      order = 2;
      parameters.resize(order);
      parameters[0] = 0.0;
      parameters[1] = 1.0;
      break;
    default:
     Assert(false,
      ExcMessage("Specified scheme is not implemented. See documentation"));
  };
  coefficients.reinit(order);
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
    timestep(desired_start_step_size),
    old_timestep(desired_start_step_size)
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
  update_coefficients();
  output = coefficients;
}

void VSIMEXMethod::set_proposed_step_size(const double &timestep)
{
  if (timestep < timestep_lower_bound)
    set_desired_next_step_size(timestep_lower_bound);
  else if (timestep > timestep_upper_bound)
    set_desired_next_step_size(timestep_upper_bound);
  else
    set_desired_next_step_size(timestep);
}

void VSIMEXMethod::update_coefficients()
{
  timestep      = get_next_step_size();
  old_timestep  = get_previous_step_size();
  omega         = timestep / old_timestep;

  switch (order)
  {
    case 1 :
      {
      static double gamma   = parameters[0];
      coefficients.alpha[0] = 1.0 / timestep;
      coefficients.alpha[1] = - 1.0 / timestep;
      coefficients.beta[0]  = 1.0;
      coefficients.gamma[0] = ( 1.0 - gamma);
      coefficients.gamma[1] = gamma;
      coefficients.phi[0]   = -1.0;
      coefficients.phi[1]   = 2.0;
      break;
      }
    case 2 :
      {
      static double gamma   = parameters[0];
      static double c       = parameters[1];
      coefficients.alpha[0] = (2.0 * gamma - 1.0) * omega * omega / 
                                (1.0 + omega) / timestep;
      coefficients.alpha[1] = ((1.0 - 2.0 * gamma) * omega - 1.0) / timestep;
      coefficients.alpha[2] = (1.0 + 2.0 * gamma * omega) / 
                                (1.0 + omega) / timestep;
      coefficients.beta[0]  = - gamma * omega;
      coefficients.beta[1]  = 1.0 + gamma * omega;
      coefficients.gamma[0] = c / 2.0;
      coefficients.gamma[1] = 1.0 - gamma - (1.0 + 1.0 / omega) * c / 2.0; 
      coefficients.gamma[2] = gamma + c / (2.0 * omega); 
      coefficients.phi[0]   = - omega;
      coefficients.phi[1]   = 1.0 + omega;
      break;
      }
    default :
     Assert(false,
      ExcMessage("Only VSIMEX of first and second order are currently implemented"));
  }
  //coefficients.output();
}

} // namespace TimeDiscretiation
} // namespace RMHD
