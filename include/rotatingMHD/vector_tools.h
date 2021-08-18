#ifndef INCLUDE_ROTATINGMHD_VECTOR_TOOLS_H_
#define INCLUDE_ROTATINGMHD_VECTOR_TOOLS_H_

#include <rotatingMHD/finite_element_field.h>

#include <deal.II/numerics/vector_tools.h>

#include <map>

namespace RMHD
{

namespace VectorTools
{

using NormType = dealii::VectorTools::NormType;

/*!
 * Computes the error of the solution specified by the finite element field with
 * respect to the given exact solution.
 */
template <int dim, typename VectorType>
std::map<NormType, double>
compute_error
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &exact_solution);

/*!
 * Same as @ref compute_error above but with a default mapping.
 */
template <int dim, typename VectorType>
std::map<NormType, double>
compute_error
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &exact_solution);

/*!
 * Interpolates the function on the support points at the finite element field
 * and applies the hanging node constraints.
 */
template <int dim, typename VectorType>
void interpolate
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector);

/*!
 * Same as @ref interpolate above but with a default mapping.
 */
template <int dim, typename VectorType>
void interpolate
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector);

/*!
 * Projects the function to the finite element field and applies the constraints
 * specified by the finite element field.
 */
template <int dim, typename VectorType>
void project
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector);

/*!
 * Same as @ref project above but with a default mapping.
 */
template <int dim, typename VectorType>
void project
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector);

}  // namespace VectorTools

}  // namespace RMHD



#endif /* INCLUDE_ROTATINGMHD_VECTOR_TOOLS_H_ */
