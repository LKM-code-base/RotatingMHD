
#ifndef INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_
#define INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_

#include <deal.II/base/discrete_time.h>
#include <iostream>
#include <vector>
namespace RMHD
{
  using namespace dealii;

namespace TimeDiscretization
{

struct VSIMEXCoefficients
{
  std::vector<double> alpha;
  std::vector<double> beta;
  std::vector<double> gamma;

  /* Forward projection parameters, where
      u_\textrm{n+1} = phi_1 * u_\textrm{n} + phi_0 * u_\textrm{n-1}
    for the variable time step case. For the constant time step
    case the parameters reduce to phi_1 = 2 and phi_0 = -1 */
  std::vector<double> phi;

  VSIMEXCoefficients(const unsigned int                 &order);
  VSIMEXCoefficients(const VSIMEXCoefficients           &data);
  VSIMEXCoefficients operator=(const VSIMEXCoefficients &data_to_copy);
  void output();
};

class VSIMEXMethod : public DiscreteTime
{
public:
  VSIMEXMethod(const unsigned int         order,
               const std::vector<double>  parameters,
               const double               start_time,
               const double               end_time,
               const double               desired_start_step_size = 0);
  
  unsigned int        get_order();
  std::vector<double> get_parameters();
  void                get_coefficients(VSIMEXCoefficients &output);
  void                update_coefficients();
  /* This class is just declared but not defined. Would you prefer to
     just have get_coefficients and update_coefficients sepparated,
     together in update_and_get_coefficients, or all three? */
  void                update_and_get_coefficients(VSIMEXCoefficients 
                                                              &output);
private:
  const unsigned int        order;
  const std::vector<double> parameters;
  VSIMEXCoefficients        coefficients;
  double                    omega;
  double                    k_n;
  double                    k_n_minus_1;
};

} // namespace TimeDiscretization
} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_TIME_DISCRETIZATION_H_ */