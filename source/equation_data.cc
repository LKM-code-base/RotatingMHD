/*
 * equation_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/equation_data.h>
#include <deal.II/base/geometric_utilities.h>

namespace RMHD
{
  using namespace dealii;

namespace EquationData
{

namespace Step35
{

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
VelocityInflowBoundaryCondition<dim>::VelocityInflowBoundaryCondition
(const double time,
 const double maximum_velocity,
 const double height)
:
Function<dim>(dim, time),
maximum_velocity(maximum_velocity),
height(height)
{}

template <int dim>
void VelocityInflowBoundaryCondition<dim>::vector_value
(const Point<dim>  &point,
 Vector<double>    &values) const
{
  values[0] = 4.0 * maximum_velocity * point(1) * ( height - point(1) )
      / ( height * height );

  for (unsigned d=1; d<dim; ++d)
    values[d] = 0.0;
}

} // namespace DFG

namespace TGV
{

template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution
(const double Re,
 const double time)
:
Function<dim>(dim, time),
Re(Re)
{}

template <int dim>
void VelocityExactSolution<dim>::vector_value
(const Point<dim>  &point,
 Vector<double>    &values) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k / Re * t);

  values[0] =   F * cos(k * x) * sin(k * y);
  values[1] = - F * sin(k * x) * cos(k * y);
}

template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::gradient
(const Point<dim>  &point,
 const unsigned int component) const
{
  Tensor<1, dim>  return_value;

  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k / Re * t);

  // The gradient has to match that of dealii, i.e. from the right.
  if (component == 0)
  {
    return_value[0] = - F * k * sin(k * x) * sin(k * y);
    return_value[1] =   F * k * cos(k * x) * cos(k * y);
  }
  else if (component == 1)
  {
    return_value[0] = - F * k * cos(k * x) * cos(k * y);
    return_value[1] =   F * k * sin(k * x) * sin(k * y);
  }

  return return_value;
}

template <int dim>
PressureExactSolution<dim>::PressureExactSolution
(const double Re,
 const double time)
:
Function<dim>(1, time),
Re(Re)
{}

template<int dim>
double PressureExactSolution<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k / Re * t);

  return (-0.25 * F * F *(cos(2. * k * x) + cos(2. * k * y)));
}

template<int dim>
Tensor<1, dim> PressureExactSolution<dim>::gradient
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  Tensor<1, dim>  return_value;
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k / Re * t);

  return_value[0] = 0.5 * F * F * k * sin(2. * k * x);
  return_value[1] = 0.5 * F * F * k * sin(2. * k * y);

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
double PressureExactSolution<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  return sin(x - y + t);
}

template<int dim>
Tensor<1, dim> PressureExactSolution<dim>::gradient
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  Tensor<1, dim>  return_value;
  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  return_value[0] =   cos(x - y + t);
  return_value[1] = - cos(x - y + t);

  return return_value;
}

template <int dim>
BodyForce<dim>::BodyForce
(const double Re,
 const double time)
:
TensorFunction<1, dim>(time),
Re(Re)
{}

template <int dim>
Tensor<1, dim> BodyForce<dim>::value(const Point<dim> &point) const
{
  // The commented out lines corresponds to the case where the convection
  // term is ignored.
  Tensor<1, dim> value;

  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  value[0] = cos(t + x - y) + sin(2.*(t + x))/2. +
              (2.*sin(t + x)*sin(t + y))/Re + sin(2.*t + x + y)
              /*cos(t + x - 1.*y) + (2.*sin(t + x)*sin(t + y))/Re
              + sin(2.*t + x + y)*/;
  value[1] = (cos(x - y) + cos(2.*t + x + y) - (Re*(2.*cos(t + x - y) +
              sin(2.*(t + y)) + 2.*sin(2.*t + x + y)))/2.)/Re
              /*(cos(x - 1.*y) + cos(2.*t + x + y) -
              1.*Re*(cos(t + x - 1.*y) + sin(2.*t + x + y)))/Re*/;

  return value;
}

} // namespace Guermond

namespace GuermondNeumannBC
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
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  values[0] = sin(x) * sin(y + t);
  values[1] = cos(x) * cos(y + t);
}

template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::gradient(
  const Point<dim>  &point,
  const unsigned int component) const
{
  Tensor<1, dim>  return_value;

  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  // The gradient has to match that of dealii, i.e. from the right.
  if (component == 0)
  {
    return_value[0] = cos(x) * sin(y + t);
    return_value[1] = sin(x) * cos(y + t);
  }
  else if (component == 1)
  {
    return_value[0] = - sin(x) * cos(y + t);
    return_value[1] = - cos(x) * sin(y + t);
  }

  return return_value;
}

template <int dim>
PressureExactSolution<dim>::PressureExactSolution(const double time)
:
Function<dim>(1, time)
{}

template<int dim>
double PressureExactSolution<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  return (cos(x) * sin(y + t));
}

template<int dim>
Tensor<1, dim> PressureExactSolution<dim>::gradient
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  Tensor<1, dim>  return_value;
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  return_value[0] =  - sin(x) * sin(y + t);
  return_value[1] = cos(x) * cos(y + t);

  return return_value;
}

template <int dim>
BodyForce<dim>::BodyForce
(const double Re,
 const double time)
:
TensorFunction<1, dim>(time),
Re(Re)
{}

template <int dim>
Tensor<1, dim> BodyForce<dim>::value(const Point<dim> &point) const
{
  // The commented out lines corresponds to the case where the convection
  // term is ignored.
  Tensor<1, dim> value;

  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  // With advection term
  value[0] = (sin(x)*(Re*(cos(x) + cos(t + y)) - 1.*(-2. + Re)*sin(t + y)))/Re;
  value[1] = (((2. + Re)*cos(x)*cos(t + y))/Re - 1.*(cos(x) + cos(t + y))*sin(t + y));

  /*
  // Without advection term
  value[0] = ((sin(x)*(Re*cos(t + y) - 1.*(-2. + Re)*sin(t + y)))/Re);
  value[1] = (cos(x)*((2. + Re)*cos(t + y) - 1.*Re*sin(t + y)))/Re;
  */

  return value;
}


} // namespace GuermondNeumannBC
namespace Couette
{

template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution
(const double t_0,
 const double Re,
 const double H,
 const double time)
:
Function<dim>(dim, time),
t_0(t_0),
Re(Re),
H(H)
{}

template <int dim>
void VelocityExactSolution<dim>::vector_value(
  const Point<dim>  &point,
  Vector<double>    &values) const
{
  const double y = point(1);

  values[0] = t_0 * Re * y / H;
  values[1] = 0.0;
}

template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::gradient(
  const Point<dim>  &/*point*/,
  const unsigned int component) const
{
  Tensor<1, dim>  gradient;

  // The gradient has to match that of dealii, i.e. from the right.
  if (component == 0)
  {
    gradient[0] = 0.0;
    gradient[1] = t_0 * Re / H;
  }
  else if (component == 1)
  {
    gradient[0] = 0.0;
    gradient[1] = 0.0;
  }

  return gradient;
}

template <int dim>
TractionVector<dim>::TractionVector
(const double t_0,
 const double time)
:
TensorFunction<1, dim>(time),
t_0(t_0)
{}

template <int dim>
Tensor<1, dim> TractionVector<dim>::value(const Point<dim> &/*point*/) const
{
  Tensor<1, dim> traction_vector;

  traction_vector[0] = t_0;
  traction_vector[1] = 0.0;

  return traction_vector;
}
} // namespace Couette
namespace ThermalTGV
{

template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution
(const double time)
:
TensorFunction<1, dim>(time)
{}

template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::value
(const Point<dim>  &point) const
{
  Tensor<1, dim>  return_value;

  const double x = point(0);
  const double y = point(1);

  return_value[0] = cos(k * x) * cos(k * y);
  return_value[1] = sin(k * x) * sin(k * y);

  return return_value;
}


template <int dim>
TemperatureExactSolution<dim>::TemperatureExactSolution
(const double Pe,
 const double time)
:
Function<dim>(1, time),
Pe(Pe)
{}

template<int dim>
double TemperatureExactSolution<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k  / Pe * t);

  return (F *(cos(k * x) * sin(k * y)));
}

template<int dim>
Tensor<1, dim> TemperatureExactSolution<dim>::gradient
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  Tensor<1, dim>  return_value;
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k  / Pe * t);

  return_value[0] = - F * k * sin(k * x) * sin(k * y);
  return_value[1] = + F * k * cos(k * x) * cos(k * y);

  return return_value;
}

template <int dim>
VelocityField<dim>::VelocityField
(const double time)
:
Function<dim>(dim, time)
{}

template <int dim>
void VelocityField<dim>::vector_value
(const Point<dim>  &point,
 Vector<double>    &values) const
{
  const double x = point(0);
  const double y = point(1);

  values[0] = cos(k * x) * cos(k * y);
  values[1] = sin(k * x) * sin(k * y);
}

} // namespace ThermalTGV

} // namespace EquationData

} // namespace RMHD


// explicit instantiation
template class RMHD::EquationData::Step35::VelocityInflowBoundaryCondition<2>;
template class RMHD::EquationData::Step35::VelocityInflowBoundaryCondition<3>;

template class RMHD::EquationData::Step35::PressureInitialCondition<2>;
template class RMHD::EquationData::Step35::PressureInitialCondition<3>;

template class RMHD::EquationData::DFG::VelocityInflowBoundaryCondition<2>;
template class RMHD::EquationData::DFG::VelocityInflowBoundaryCondition<3>;

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

template class RMHD::EquationData::GuermondNeumannBC::VelocityExactSolution<2>;
template class RMHD::EquationData::GuermondNeumannBC::VelocityExactSolution<3>;

template class RMHD::EquationData::GuermondNeumannBC::PressureExactSolution<2>;
template class RMHD::EquationData::GuermondNeumannBC::PressureExactSolution<3>;

template class RMHD::EquationData::GuermondNeumannBC::BodyForce<2>;
template class RMHD::EquationData::GuermondNeumannBC::BodyForce<3>;

template class RMHD::EquationData::Couette::VelocityExactSolution<2>;
template class RMHD::EquationData::Couette::VelocityExactSolution<3>;

template class RMHD::EquationData::Couette::TractionVector<2>;
template class RMHD::EquationData::Couette::TractionVector<3>;

template class RMHD::EquationData::ThermalTGV::VelocityExactSolution<2>;
template class RMHD::EquationData::ThermalTGV::VelocityExactSolution<3>;

template class RMHD::EquationData::ThermalTGV::TemperatureExactSolution<2>;
template class RMHD::EquationData::ThermalTGV::TemperatureExactSolution<3>;

template class RMHD::EquationData::ThermalTGV::VelocityField<2>;
template class RMHD::EquationData::ThermalTGV::VelocityField<3>;
