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
                                        const Point<dim>  &point,
                                        Vector<double>    &values) const
{
  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  values[0] =  exp(-2.0/Re*t)*cos(x)*sin(y);
  values[1] = -exp(-2.0/Re*t)*sin(x)*cos(y);
}

template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::gradient(
  const Point<dim>  &point,
  const unsigned int component) const
{
  Tensor<1, dim>  return_value;

  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  // The gradient has to match that of dealii, i.e. from the right.
  if (component == 0)
  {
    return_value[0] = - exp(-2.0/Re*t) * sin(x) * sin(y);
    return_value[1] =   exp(-2.0/Re*t) * cos(x) * cos(y);
  }
  else if (component == 1)
  {
    return_value[0] = - exp(-2.0/Re*t) * cos(x) * cos(y);
    return_value[1] =   exp(-2.0/Re*t) * sin(x) * sin(y);
  }

  return return_value;
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
                                    const Point<dim> &point,
                                    const unsigned int component) const
{
  (void)component;
  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  return -0.25*exp(-4.0/Re*t)*(cos(2.0*x)+cos(2.0*y));
}

template<int dim>
Tensor<1, dim> PressureExactSolution<dim>::gradient(
  const Point<dim> &point,
  const unsigned int component) const
{
  (void)component;
  Tensor<1, dim>  return_value;
  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  return_value[0] = 0.5 * exp(-4.0 / Re * t) * sin(2.0 * x);
  return_value[1] = 0.5 * exp(-4.0 / Re * t) * sin(2.0 * y);

  return return_value;
}
} // namespace TGV

namespace Guermond
{

template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution(const double time)
: 
Function<dim>(dim, time)
{}

template <int dim>
void VelocityExactSolution<dim>::vector_value(
                                        const Point<dim>  &point,
                                        Vector<double>    &values) const
{
  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  values[0] = sin(x + t) * sin(y + t);
  values[1] = cos(x + t) * cos(y + t);
}

template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::gradient(
  const Point<dim>  &point,
  const unsigned int component) const
{
  Tensor<1, dim>  return_value;

  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  // The gradient has to match that of dealii, i.e. from the right.
  if (component == 0)
  {
    return_value[0] = cos(x + t) * sin(y + t);
    return_value[1] = sin(x + t) * cos(y + t);
  }
  else if (component == 1)
  {
    return_value[0] = - sin(x + t) * cos(y + t);
    return_value[1] = - cos(x + t) * sin(y + t);
  }

  return return_value;
}

template <int dim>
PressureExactSolution<dim>::PressureExactSolution(const double time)
:
Function<dim>(1, time)
{}

template<int dim>
double PressureExactSolution<dim>::value(
                                    const Point<dim> &point,
                                    const unsigned int component) const
{
  (void)component;
  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  return sin(x - y + t);
}

template<int dim>
Tensor<1, dim> PressureExactSolution<dim>::gradient(
  const Point<dim> &point,
  const unsigned int component) const
{
  (void)component;
  Tensor<1, dim>  return_value;
  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  return_value[0] =   cos(x - y + t);
  return_value[1] = - cos(x - y + t);

  return return_value;
}

template <int dim>
BodyForce<dim>::BodyForce(const double &Re, const double time)
: 
Function<dim>(dim, time),
Re(Re)
{}

template <int dim>
void BodyForce<dim>::vector_value(const Point<dim>  &point,
                                  Vector<double>    &values) const
{
  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  values[0] = cos(t + x - y) + sin(2.*(t + x))/2. + 
              (2.*sin(t + x)*sin(t + y))/Re + sin(2.*t + x + y);
  values[1] = (cos(x - y) + cos(2.*t + x + y) - (Re*(2.*cos(t + x - y) + 
              sin(2.*(t + y)) + 2.*sin(2.*t + x + y)))/2.)/Re;
}

} // namespace Guermond

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

template class RMHD::EquationData::Guermond::VelocityExactSolution<2>;
template class RMHD::EquationData::Guermond::VelocityExactSolution<3>;

template class RMHD::EquationData::Guermond::PressureExactSolution<2>;
template class RMHD::EquationData::Guermond::PressureExactSolution<3>;

template class RMHD::EquationData::Guermond::BodyForce<2>;
template class RMHD::EquationData::Guermond::BodyForce<3>;