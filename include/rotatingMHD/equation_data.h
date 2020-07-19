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

namespace Step35
{

  using namespace dealii;

namespace EquationData
{

template <int dim>
class VelocityIC : public Function<dim>
{

public:
  VelocityIC(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;
};

template <int dim>
class VelocityInflowBC : public Function<dim>
{
public:
  VelocityInflowBC(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;
};

template <int dim>
class PressureIC : public Function<dim>
{
public:
  PressureIC(const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;
};

} // namespace EquationData

} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_EQUATION_DATA_H_ */
