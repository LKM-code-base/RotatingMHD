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

template <int dim, typename VectorType>
std::map<NormType, double>
compute_error
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &exact_solution);

template <int dim, typename VectorType>
std::map<NormType, double>
compute_error
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &exact_solution);

template <int dim, typename VectorType>
void interpolate
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector);

template <int dim, typename VectorType>
void interpolate
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector);

}  // namespace VectorTools

}  // namespace RMHD



#endif /* INCLUDE_ROTATINGMHD_VECTOR_TOOLS_H_ */
