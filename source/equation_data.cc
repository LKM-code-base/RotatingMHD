/*
 * equation_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/equation_data.h>
#include <cmath>

namespace RMHD
{
  using namespace dealii;

namespace EquationData
{

namespace Step35
{

template <int dim>
VelocityInitialCondition<dim>::VelocityInitialCondition(const double time)
:
Function<dim>(dim, time)
{}

template <int dim>
void VelocityInitialCondition<dim>::vector_value
(const Point<dim>  &/* point */,
 Vector<double>    &values) const
{
    values[0] = 0.0;
    values[1] = 0.0;
}

template <int dim>
VelocityInflowBoundaryCondition<dim>::VelocityInflowBoundaryCondition
(const double time)
:
Function<dim>(dim, time)
{}

template <int dim>
void VelocityInflowBoundaryCondition<dim>::vector_value
(const Point<dim>  &point,
 Vector<double>    &values) const
{
  const double Um = 1.5;
  const double H  = 4.1;

  values[0] = 4.0 * Um * point(1) * ( H - point(1) ) / ( H * H );
  values[1] = 0.0;
}

template <int dim>
PressureInitialCondition<dim>::PressureInitialCondition(const double time)
:
Function<dim>(1, time)
{}

template<int dim>
double PressureInitialCondition<dim>::value
(const Point<dim> &p,
 const unsigned int /* component */) const
{
  return (25.0 - p(0)) ;
}

} // namespace Step35

namespace DFG
{

template <int dim>
VelocityInitialCondition<dim>::VelocityInitialCondition(const double time)
:
Function<dim>(dim, time)
{}

template <int dim>
void VelocityInitialCondition<dim>::vector_value
(const Point<dim>  &/* point */,
 Vector<double>    &values) const
{
    values[0] = 0.0;
    values[1] = 0.0;
}

template <int dim>
VelocityInflowBoundaryCondition<dim>::VelocityInflowBoundaryCondition
(const double time)
:
Function<dim>(dim, time)
{}

template <int dim>
void VelocityInflowBoundaryCondition<dim>::vector_value
(const Point<dim>  &point,
 Vector<double>    &values) const
{
  const double Um = 1.5;
  const double H  = 4.1;

  values[0] = 4.0 * Um * point(1) * ( H - point(1) ) / ( H * H );
  values[1] = 0.0;
}

template <int dim>
PressureInitialCondition<dim>::PressureInitialCondition(const double time)
:
Function<dim>(1, time)
{}

template<int dim>
double PressureInitialCondition<dim>::value(const Point<dim> &/* point */,
                                            const unsigned int /* component */) const
{
  return (0.0);
}

} // namespace DFG

namespace TGV
{

template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution(const double &Re,
                                                  const double time)
: 
Function<dim>(dim, time),
Re(Re)
{}

template <int dim>
void VelocityExactSolution<dim>::vector_value(
                                        const Point<dim>  &p,
                                        Vector<double>    &values) const
{
    double t = this->get_time();
    values[0] = exp(-2.0/Re*t)*cos(p(0))*sin(p(1));
    values[1] = -exp(-2.0/Re*t)*sin(p(0))*cos(p(1));
}

template <int dim>
PressureExactSolution<dim>::PressureExactSolution(const double &Re,
                                                  const double time)
:
Function<dim>(1, time),
Re(Re)
{}

template<int dim>
double PressureExactSolution<dim>::value(
                                    const Point<dim> &p,
                                    const unsigned int component) const
{
  (void)component;
  double t = this->get_time();
  return -0.25*exp(-4.0/Re*t)*(cos(2.0*p(0))+cos(2.0*p(1)));
}
} // namespace TGV

} // namespace EquationData

} // namespace RMHD


// explicit instantiation

template class RMHD::EquationData::Step35::VelocityInitialCondition<2>;
template class RMHD::EquationData::Step35::VelocityInitialCondition<3>;

template class RMHD::EquationData::Step35::VelocityInflowBoundaryCondition<2>;
template class RMHD::EquationData::Step35::VelocityInflowBoundaryCondition<3>;

template class RMHD::EquationData::Step35::PressureInitialCondition<2>;
template class RMHD::EquationData::Step35::PressureInitialCondition<3>;

template class RMHD::EquationData::DFG::VelocityInitialCondition<2>;
template class RMHD::EquationData::DFG::VelocityInitialCondition<3>;

template class RMHD::EquationData::DFG::VelocityInflowBoundaryCondition<2>;
template class RMHD::EquationData::DFG::VelocityInflowBoundaryCondition<3>;

template class RMHD::EquationData::DFG::PressureInitialCondition<2>;
template class RMHD::EquationData::DFG::PressureInitialCondition<3>;

template class RMHD::EquationData::TGV::VelocityExactSolution<2>;
template class RMHD::EquationData::TGV::VelocityExactSolution<3>;

template class RMHD::EquationData::TGV::PressureExactSolution<2>;
template class RMHD::EquationData::TGV::PressureExactSolution<3>;
