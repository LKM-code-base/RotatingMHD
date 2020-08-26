#ifndef INCLUDE_ROTATINGMHD_AUXILIARY_FUNCTIONS_H_
#define INCLUDE_ROTATINGMHD_AUXILIARY_FUNCTIONS_H_

#include <rotatingMHD/entities_structs.h>

using namespace dealii;

template <int dim>
void mpi_point_value(
                  const RMHD::Entities::EntityBase<dim> &entity,
                  const Point<dim>                &point,
                  double                          &scalar_point_value);

template <int dim>
void mpi_point_value(
                  const RMHD::Entities::EntityBase<dim> &entity,
                  const Point<dim>                &point,
                  Vector<double>                  &vector_point_value);

#endif /*INCLUDE_ROTATINGMHD_AUXILIARY_FUNCTIONS_H_*/