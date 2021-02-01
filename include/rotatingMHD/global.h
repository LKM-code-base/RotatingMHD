/*
 * global.h
 *
 *  Created on: Aug 29, 2020
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_GLOBAL_H_
#define INCLUDE_ROTATINGMHD_GLOBAL_H_

#include <deal.II/lac/generic_linear_algebra.h>

/*
 * Uncomment the following line for forcing the usage of the Trilinos library
 */
#define FORCE_USE_OF_TRILINOS

namespace RMHD
{

/*!
 * A namespace that contains Typedefs for classes used for the linear algebra.
 */
namespace LinearAlgebra
{

#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
  #define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
  #error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif

} // namespace LinearAlgebra

}  // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_GLOBAL_H_ */
