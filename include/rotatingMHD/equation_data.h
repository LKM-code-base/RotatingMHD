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

  virtual double divergence(
    const Point<dim>  &point) const;

  virtual void divergence_list(
    const std::vector<Point<dim>> &points,
    std::vector<double>           &values) const;
};

namespace Step35
{
template <int dim>
class VelocityInitialCondition : public Function<dim>
{
public:
  VelocityInitialCondition(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;
};

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
template <int dim>
class VelocityInitialCondition : public Function<dim>
{
public:
  VelocityInitialCondition(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;
};

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

} // namespace EquationData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_EQUATION_DATA_H_ */
