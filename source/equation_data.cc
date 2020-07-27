/*
 * equation_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/equation_data.h>

namespace Step35
{
  using namespace dealii;

namespace EquationData
{

template <int dim>
VelocityInitialCondition<dim>::VelocityInitialCondition(const double time)
  : Function<dim>(dim, time)
{}

template <int dim>
void VelocityInitialCondition<dim>::vector_value(
                                        const Point<dim>  &p,
                                        Vector<double>    &values) const
{
    (void)p;
    values[0] = 0.0;
    values[1] = 0.0;
}

template <int dim>
VelocityInflowBoundaryCondition<dim>::VelocityInflowBoundaryCondition(
                                                      const double time)
: Function<dim>(dim, time)
{}

template <int dim>
void VelocityInflowBoundaryCondition<dim>::vector_value(
                                        const Point<dim>  &p,
                                        Vector<double>    &values) const
{
  const double Um = 1.5;
  const double H  = 4.1;

  values[0] = 4.0 * Um * p(1) * ( H - p(1) ) / ( H * H );
  values[1] = 0.0;
}

template <int dim>
PressureInitialCondition<dim>::PressureInitialCondition(const double time)
: Function<dim>(1, time)
{}

template<int dim>
double PressureInitialCondition<dim>::value(
                                    const Point<dim> &p,
                                    const unsigned int component) const
{
  (void)component;
  return 25.0 - p(0);
}

} // namespace EquationData

} // namespace Step35


// explicit instantiation

template class Step35::EquationData::VelocityInitialCondition<2>;
template class Step35::EquationData::VelocityInitialCondition<3>;

template class Step35::EquationData::VelocityInflowBoundaryCondition<2>;
template class Step35::EquationData::VelocityInflowBoundaryCondition<3>;

template class Step35::EquationData::PressureInitialCondition<2>;
template class Step35::EquationData::PressureInitialCondition<3>;
