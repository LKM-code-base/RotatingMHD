/*
 * angular_velocity.h
 *
 *  Created on: Aug 18, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_ANGULAR_VELOCITY_H_
#define INCLUDE_ROTATINGMHD_ANGULAR_VELOCITY_H_

#include <deal.II/base/function_time.h>
#include <deal.II/fe/fe_values.h>

namespace RMHD
{

template <int dim>
class AngularVelocity : public dealii::FunctionTime<double>
{
public:
  AngularVelocity(const double time = 0);

  using value_type = typename dealii::FEValuesViews::Vector<dim>::curl_type;

  virtual value_type  value() const;
};

}  // namespace RMHD



#endif /* INCLUDE_ROTATINGMHD_ANGULAR_VELOCITY_H_ */
