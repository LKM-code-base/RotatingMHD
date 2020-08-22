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

#include <deal.II/lac/vector.h>

namespace RMHD
{

  using namespace dealii;

namespace EquationData
{

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
} // namespace EquationData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_EQUATION_DATA_H_ */
