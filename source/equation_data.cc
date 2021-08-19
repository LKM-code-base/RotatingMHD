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

namespace GuermondNeumannBC
{


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
