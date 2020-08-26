#ifndef INCLUDE_ROTATINGMHD_AUXILIARY_FUNCTIONS_H_
#define INCLUDE_ROTATINGMHD_AUXILIARY_FUNCTIONS_H_

#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/function.h>


namespace RMHD
{

  using namespace dealii;

template <int dim>
class Problem 
{
public:
  Problem()
  {}

protected:
  void set_initial_conditions(
                        Entities::EntityBase<dim>         &entity,
                        Function<dim>                     &function,
                        TimeDiscretization::VSIMEXMethod  &time_stepping);
  void mpi_point_value(
                  const Entities::EntityBase<dim> &entity,
                  const Point<dim>                &point,
                  double                          &scalar_point_value);
  void mpi_point_value(
                  const Entities::EntityBase<dim> &entity,
                  const Point<dim>                &point,
                  Vector<double>                  &vector_point_value);

};

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_AUXILIARY_FUNCTIONS_H_*/