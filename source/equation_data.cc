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



template <int dim>
VectorFunction<dim>::VectorFunction(const double time)
:
TensorFunction<1, dim>(time)
{}



template <int dim>
double VectorFunction<dim>::divergence(const Point<dim>  &/* point */) const
{
  return (0.0);
}



template <int dim>
void VectorFunction<dim>::divergence_list(
  const std::vector<Point<dim>> &points,
  std::vector<double>           &values) const
{
  for (unsigned int i = 0; i < points.size(); ++i)
    values[i] = divergence(points[i]);
}



template <int dim>
CurlType<dim> VectorFunction<dim>::curl(const Point<dim>  &/* point */) const
{
  CurlType<dim> value;

  if constexpr(dim == 2)
    value[0] = 0.0;
  else if constexpr(dim ==3)
  {
    value[0] = 0.0;
    value[1] = 0.0;
    value[2] = 0.0;
  }

  return (value);
}



template <int dim>
void VectorFunction<dim>::curl_list(
  const std::vector<Point<dim>> &points,
  std::vector<CurlType<dim>>    &values) const
{
  for (unsigned int i = 0; i < points.size(); ++i)
    values[i] = curl(points[i]);
}



template <int dim>
AngularVelocity<dim>::AngularVelocity(const double time)
:
VectorFunction<dim>(time)
{}



template <int dim>
CurlType<dim> AngularVelocity<dim>::rotation() const
{
  CurlType<dim> value;

  if constexpr(dim == 2)
    value[0] = 0.0;
  else if constexpr(dim ==3)
  {
    value[0] = 0.0;
    value[1] = 0.0;
    value[2] = 0.0;
  }

  return (value);
}



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
RMHD::EquationData::VectorFunction<dim>(time),
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

template <int dim>
double BodyForce<dim>::divergence(const Point<dim>  &point) const
{
  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  return cos(2.*(t + x)) + cos(2.*t + x + y) - 1.*sin(t + x - 1.*y) +
         (2.*cos(t + x)*sin(t + y))/Re + (sin(x - 1.*y) -
         0.5*Re*(2.*cos(2.*(t + y)) + 2.*cos(2.*t + x + y) +
         2.*sin(t + x - 1.*y)) - 1.*sin(2.*t + x + y))/Re
         /*cos(2.*t + x + y) - 1.*sin(t + x - 1.*y) +
         (2.*cos(t + x)*sin(t + y))/Re + (sin(x - 1.*y) -
         1.*Re*(cos(2.*t + x + y) + sin(t + x - 1.*y)) -
         1.*sin(2.*t + x + y))/Re*/;
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
RMHD::EquationData::VectorFunction<dim>(time),
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

template <int dim>
double BodyForce<dim>::divergence(const Point<dim>  &point) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  // With advection term
  return (cos(2.*x) - 1.*cos(2.*(t + y)) - 2.*cos(x)*sin(t + y));

  /*
  // Without advection term
  return (-2.*cos(x)*sin(t + y));
  */
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

namespace MIT
{

template <int dim>
TemperatureBoundaryCondition<dim>::TemperatureBoundaryCondition
(const double time)
:
Function<dim>(1, time)
{}

template<int dim>
double TemperatureBoundaryCondition<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double t = this->get_time();
  const double x = point(0);

  const double sign = ( x < 0.5 ) ? 1.0 : -1.0;

  return ( sign * 0.5 * (1.0 - exp(- beta * t)));
}


template <int dim>
GravityUnitVector<dim>::GravityUnitVector
(const double time)
:
RMHD::EquationData::VectorFunction<dim>(time)
{}

template <int dim>
Tensor<1, dim> GravityUnitVector<dim>::value(const Point<dim> &/*point*/) const
{
  Tensor<1, dim> value;

  value[0] = 0.0;
  value[1] = -1.0;

  return value;
}

} // namespace MIT



namespace Christensen
{



template <int dim>
TemperatureInitialCondition<dim>::TemperatureInitialCondition
(const double r_i,
 const double r_o,
 const double A,
 const double time)
:
Function<dim>(1, time),
r_i(r_i),
r_o(r_o),
A(A)
{
  Assert(r_o > r_i, ExcMessage("The outer radius has to be greater then the inner radius"))
}

template<int dim>
double TemperatureInitialCondition<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  double temperature;

  const std::array<double, dim> spherical_coordinates = GeometricUtilities::Coordinates::to_spherical(point);

  // Radius
  const double r        = spherical_coordinates[0];
  // Azimuthal angle
  const double phi      = spherical_coordinates[1];
  // Polar angle
  double theta;
  if constexpr(dim == 2)
    theta = M_PI_2;
  else if constexpr(dim == 3)
    theta = spherical_coordinates[2];

  const double x_0    = 2. * r - r_i - r_o;

  temperature = r_o * r_i / r
                - r_i
                + 210. * A / std::sqrt(17920. * M_PI) *
                  (1. - 3. * x_0 * x_0 + 3. * std::pow(x_0, 4) - std::pow(x_0,6)) *
                  pow(std::sin(theta), 4) *
                  std::cos(4. * phi);

  return temperature;
}




template <int dim>
TemperatureBoundaryCondition<dim>::TemperatureBoundaryCondition
(const double r_i,
 const double r_o,
 const double time)
:
Function<dim>(1, time),
r_i(r_i),
r_o(r_o)
{
  Assert(r_o > r_i, ExcMessage("The outer radius has to be greater then the inner radius"))
}



template<int dim>
double TemperatureBoundaryCondition<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double r = point.norm();

  double value = (r > 0.5*(r_i + r_o)) ? 0.0 : 1.0;

  return (value);
}



template <int dim>
GravityVector<dim>::GravityVector
(const double r_o,
 const double time)
:
RMHD::EquationData::VectorFunction<dim>(time),
r_o(r_o)
{}



template <int dim>
Tensor<1, dim> GravityVector<dim>::value(const Point<dim> &point) const
{
  Tensor<1, dim> value;

  const double x    = point(0);
  const double y    = point(1);

  value[0]    = - x / r_o;
  value[1]    = - y / r_o;

  if constexpr(dim == 3)
  {
    const double z  = point(2);
    value[2]        = - z / r_o;
  }


  return value;
}



template <int dim>
AngularVelocity<dim>::AngularVelocity
(const double time)
:
RMHD::EquationData::AngularVelocity<dim>(time)
{}



template <int dim>
CurlType<dim> AngularVelocity<dim>::rotation() const
{
  CurlType<dim> value;

  if constexpr(dim == 2)
    value[0] = 1.;
  else if constexpr(dim == 3)
  {
    value[0] = 0.;
    value[1] = 0.;
    value[2] = 1.;
  }

  return value;
}



} // namespace Christensen


} // namespace EquationData

} // namespace RMHD


// explicit instantiation

template class RMHD::EquationData::VectorFunction<2>;
template class RMHD::EquationData::VectorFunction<3>;

template class RMHD::EquationData::AngularVelocity<2>;
template class RMHD::EquationData::AngularVelocity<3>;

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

template class RMHD::EquationData::MIT::TemperatureBoundaryCondition<2>;
template class RMHD::EquationData::MIT::TemperatureBoundaryCondition<3>;

template class RMHD::EquationData::MIT::GravityUnitVector<2>;
template class RMHD::EquationData::MIT::GravityUnitVector<3>;

template class RMHD::EquationData::Christensen::TemperatureInitialCondition<2>;
template class RMHD::EquationData::Christensen::TemperatureInitialCondition<3>;

template class RMHD::EquationData::Christensen::TemperatureBoundaryCondition<2>;
template class RMHD::EquationData::Christensen::TemperatureBoundaryCondition<3>;

template class RMHD::EquationData::Christensen::GravityVector<2>;
template class RMHD::EquationData::Christensen::GravityVector<3>;

template class RMHD::EquationData::Christensen::AngularVelocity<2>;
template class RMHD::EquationData::Christensen::AngularVelocity<3>;

