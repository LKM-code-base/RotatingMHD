/*
 * equation_data.h
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_EQUATION_DATA_H_
#define INCLUDE_ROTATINGMHD_EQUATION_DATA_H_

#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/lac/vector.h>

#include <cmath>

namespace RMHD
{

using namespace dealii;

namespace EquationData
{

template <int dim>
class BodyForce : public TensorFunction<1, dim>
{

public:

  BodyForce(const double time = 0);

  virtual double divergence(const Point<dim>  &point) const;

  virtual void divergence_list(const std::vector<Point<dim>> &points,
                               std::vector<double>           &values) const;
};

namespace Step35
{

template<int dim>
using VelocityInitialCondition = Functions::ZeroFunction<dim>;

template <int dim>
class VelocityInflowBoundaryCondition : public Function<dim>
{
public:
  VelocityInflowBoundaryCondition(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;
};

template <int dim>
class PressureInitialCondition : public Function<dim>
{
public:
  PressureInitialCondition(const double time = 0);

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;
};
} // namespace Step35

namespace DFG
{

/*!
 * @class VelocityInflowBoundaryCondition
 *
 * @brief The velocity profile at the inlet of the channel.
 *
 * @details The velocity profile is given by the following function
 * \f[
 * \bs{v}(y)= v_x(y) \ex= v_0 \left(\frac{2}{H}\right)^2 y (H-y) \ex \,,
 * \f]
 *
 * where \f$ H \f$ denotes the height of the channel and \f$ v_0 \f$ is the
 * maximum velocity in the middle of the channel. Typically, the maximum velocity is
 * chosen as \f$ v_0=\frac{3}{2} \f$ such that the mean velocity is unity.
 *
 */
template <int dim>
class VelocityInflowBoundaryCondition : public Function<dim>
{

public:
  /*!
   * Default constructor.
   */
  VelocityInflowBoundaryCondition(const double time = 0,
                                  const double maximum_velocity = 1.5,
                                  const double height = 4.1);

  /*!
   * Overloaded method evaluating the function.
   */
  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;

private:

  /*!
   * The maximum velocity at the middle of the channel.
   */
  const double maximum_velocity;

  /*!
   * The height of the channel.
   */
  const double height;

};


template<int dim>
using VelocityInitialCondition = Functions::ZeroFunction<dim>;

template<int dim>
using PressureInitialCondition = Functions::ZeroFunction<dim>;

} // namespace DFG

namespace TGV
{
template <int dim>
class VelocityExactSolution : public Function<dim>
{
public:
  VelocityExactSolution(const double Re,
                        const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int component) const;

private:
  /*!
   * @brief The Reynolds number.
   */
  const double Re;

  /*!
   * @brief The wave number.
   */
  const double k = 2. * M_PI;
};

template <int dim>
class PressureExactSolution : public Function<dim>
{
public:
  PressureExactSolution(const double Re,
                        const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int = 0) const;

private:
  /*!
   * @brief The Reynolds number.
   */
  const double Re;

  /*!
   * @brief The wave number.
   */
  const double k = 2. * M_PI;
};
} // namespace TGV

namespace Guermond
{
template <int dim>
class VelocityExactSolution : public Function<dim>
{
public:
  VelocityExactSolution(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int component) const;
};

template <int dim>
class PressureExactSolution : public Function<dim>
{
public:
  PressureExactSolution(const double time = 0);

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int = 0) const;
};

template <int dim>
class BodyForce: public RMHD::EquationData::BodyForce<dim>
{
public:
  BodyForce(const double Re,
            const double time = 0);

  virtual Tensor<1, dim> value(
    const Point<dim>  &point) const override;

  virtual double divergence(const Point<dim>  &point) const override;

  const double Re;
};

} // namespace Guermond

namespace GuermondNeumannBC
{
template <int dim>
class VelocityExactSolution : public Function<dim>
{
public:
  VelocityExactSolution(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int component) const;
};

template <int dim>
class PressureExactSolution : public Function<dim>
{
public:
  PressureExactSolution(const double time = 0);

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int = 0) const;
};

template <int dim>
class BodyForce: public RMHD::EquationData::BodyForce<dim>
{
public:
  BodyForce(const double Re,
            const double time = 0);

  virtual Tensor<1, dim> value(
    const Point<dim>  &point) const override;

  virtual double divergence(const Point<dim>  &point) const override;

  const double Re;
};

} // namespace GuermondNeumannBC

namespace Couette
{

/*!
 * @class VelocityExactSolution
 * @brief The velocity's exact solution for the Couette flow, where the
 * displacement of the top plate is driven by a traction vector
 * @details It is given by
 * \f[ \bs{u} = t_0 \Reynolds \dfrac{y}{H} \bs{e}_\mathrm{x}, \f]
 * where \f$ t_0 \f$, \f$ \Reynolds \f$, \f$ H \f$, \f$ y \f$ and
 * \f$ \bs{e}_\mathrm{x} \f$ are the traction vector magnitude, the
 * Reynolds number, the height of the channel, the \f$ y \f$-component
 * of the position vector and the unit vector in the \f$ x \f$-direction.
 */
template <int dim>
class VelocityExactSolution : public Function<dim>
{
public:
  VelocityExactSolution(const double t_0,
                        const double Re,
                        const double H = 1.0,
                        const double time = 0.0);

  virtual void vector_value(
    const Point<dim>  &p,
    Vector<double>    &values) const override;

  virtual Tensor<1, dim> gradient(
    const Point<dim> &point,
    const unsigned int component) const override;

private:
  /*!
   * @brief The magnitude of the applied traction vector.
   */
  const double t_0;

  /*!
   * @brief The Reynodls number.
   */
  const double Re;

  /*!
   * @brief The height of the channel.
   */
  const double H;
};

/*!
 * @class TractionVector
 * @brief The traction vector applied on the top plate of the Couette
 * Flow
 * @details It is given by \f[ \bs{t} = t_0 \bs{e}_\mathrm{x}, \f]
 * where \f$ t_0 \f$ and \f$ \bs{e}_\mathrm{x} \f$ are the
 * magnitude of the traction and the unit vector in the \f$ x \f$-direction.
 */
template <int dim>
class TractionVector : public TensorFunction<1,dim>
{
public:
  TractionVector(const double t_0, const double time = 0.);

  virtual Tensor<1, dim> value(const Point<dim> &point) const override;

private:
  /*!
   * @brief The magnitude of the applied traction vector.
   */
  const double t_0;
};

} // namespace Couette
namespace ThermalTGV
{
template <int dim>
class VelocityExactSolution : public TensorFunction<1, dim>
{
public:
  VelocityExactSolution(const double time = 0);

  virtual Tensor<1, dim> value(const Point<dim>  &p) const override;

private:
  /*!
   * @brief The wave number.
   */
  const double k = 2. * M_PI;
};

template <int dim>
class TemperatureExactSolution : public Function<dim>
{
public:
  TemperatureExactSolution(const double Pe,
                           const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int = 0) const override;

private:
  /*!
   * @brief The Peclet number.
   */
  const double Pe;

  /*!
   * @brief The wave number.
   */
  const double k = 2. * M_PI;
};

template <int dim>
class VelocityField : public Function<dim>
{
public:
  VelocityField(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;
private:

  /*!
   * @brief The wave number.
   */
  const double k = 2. * M_PI;
};

} // namespace ThermalTGV


namespace MIT
{

template <int dim>
class TemperatureBoundaryCondition : public Function<dim>
{
public:
  TemperatureBoundaryCondition(const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  /*!
   * @brief The exponential coefficient.
   */
  const double beta = 10.;
};

template <int dim>
class GravityUnitVector: public RMHD::EquationData::BodyForce<dim>
{
public:
  GravityUnitVector(const double time = 0);

  virtual Tensor<1, dim> value(
    const Point<dim>  &point) const override;
};

}

} // namespace EquationData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_EQUATION_DATA_H_ */
