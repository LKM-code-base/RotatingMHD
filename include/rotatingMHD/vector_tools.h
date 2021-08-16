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

template<int dim>
std::map<NormType, double>
compute_error
(const Mapping<dim>                 &external_mapping,
 const Entities::FE_FieldBase<dim>  &fe_field,
 const Function<dim>                &exact_solution);

template<int dim>
std::map<NormType, double>
compute_error
(const Entities::FE_FieldBase<dim>  &fe_field,
 const Function<dim>                &exact_solution);


}  // namespace VectorTools

}  // namespace RMHD



#endif /* INCLUDE_ROTATINGMHD_VECTOR_TOOLS_H_ */
