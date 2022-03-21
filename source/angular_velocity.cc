/*
 * angular_velocity.cc
 *
 *  Created on: Aug 18, 2021
 *      Author: sg
 */

#include <rotatingMHD/angular_velocity.h>

namespace RMHD
{

template <int dim>
AngularVelocity<dim>::AngularVelocity(const double time)
:
FunctionTime<double>(time)
{}


template <int dim>
typename AngularVelocity<dim>::value_type
AngularVelocity<dim>::value() const
{
  value_type  value;
  value = 0;
  return (value);
}

}  // namespace RMHD

// explicit instantiation
template class RMHD::AngularVelocity<2>;
template class RMHD::AngularVelocity<3>;
