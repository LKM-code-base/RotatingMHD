#include <rotatingMHD/vector_tools.h>

namespace RMHD
{

namespace VectorTools
{


template <int dim, typename VectorType>
std::map<NormType, double>
compute_error
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &exact_solution)
{
  const Triangulation<dim> &tria{fe_field.get_triangulation()};
  const DoFHandler<dim>    &dof_handler{fe_field.get_dof_handler()};

  const unsigned int        fe_degree{fe_field.fe_degree()};
  const unsigned int        n_components{fe_field.n_components()};

  AssertThrow(n_components == exact_solution.n_components,
              ExcDimensionMismatch(n_components, exact_solution.n_components));

  Vector<double>  cellwise_error(tria.n_active_cells());

  auto compute_error
  = [&mapping, &tria, &cellwise_error, &dof_handler, &fe_field]
     (const Quadrature<dim>&quadrature,
      const Function<dim>  &exact_solution,
      const NormType        norm_type)
  ->
  double
  {
    dealii::VectorTools::integrate_difference(mapping,
                                              dof_handler,
                                              fe_field.solution,
                                              exact_solution,
                                              cellwise_error,
                                              quadrature,
                                              norm_type);
    return (dealii::VectorTools::compute_global_error(tria,
                                                      cellwise_error,
                                                      norm_type));
  };

  const QGauss<dim> quadrature_formula(fe_degree + 1);

  NormType norm_type;

  std::map<NormType, double> error_map;

  norm_type = NormType::L2_norm;
  error_map[norm_type] = compute_error(quadrature_formula,
                                       exact_solution,
                                       norm_type);

  norm_type = NormType::H1_norm;
  error_map[norm_type] = compute_error(quadrature_formula,
                                       exact_solution,
                                       norm_type);

  const QTrapez<1>     trapezoidal_rule;
  const QIterated<dim> linfty_quadrature_formula(trapezoidal_rule,
                                                 fe_degree);

  norm_type = NormType::Linfty_norm;
  error_map[norm_type] = compute_error(linfty_quadrature_formula,
                                       exact_solution,
                                       norm_type);
  return (error_map);
}



template <int dim, typename VectorType>
std::map<NormType, double>
compute_error
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &exact_solution)
{
  return (compute_error(MappingQ1<dim>(), fe_field, exact_solution));
}


template <int dim, typename VectorType>
void interpolate
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                               &vector)
{
  Assert(function.n_components == fe_field.n_components(),
         ExcMessage("The number of components of the function does not those "
                    "of the entity"));

  VectorType  tmp_vector(fe_field.distributed_vector);

  dealii::VectorTools::interpolate(mapping,
                                   fe_field.get_dof_handler(),
                                   function,
                                   tmp_vector);
  fe_field.get_constraints().distribute(tmp_vector);

  vector = tmp_vector;
}



template <int dim, typename VectorType>
void interpolate
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                               &vector)
{
  interpolate(MappingQ1<dim>(), fe_field, function, vector);
}


template <int dim, typename VectorType>
void project
(const Mapping<dim>                             &mapping,
 const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector)
{
  Assert(function.n_components == fe_field.n_components(),
         ExcMessage("The number of components of the function does not those "
                    "of the entity"));

  VectorType  tmp_vector(fe_field.distributed_vector);

  dealii::VectorTools::project(mapping,
                               fe_field.get_dof_handler(),
                               fe_field.get_constraints(),
                               QGauss<dim>(fe_field.fe_degree() + 1),
                               function,
                               tmp_vector);
  vector = tmp_vector;
}


template <int dim, typename VectorType>
void project
(const Entities::FE_FieldBase<dim, VectorType>  &fe_field,
 const Function<dim>                            &function,
 VectorType                                     &vector)
{
  project(MappingQ1<dim>(), fe_field, function, vector);
}


}  // namespace VectorTools

}  // namespace RMHD

// explicit instantiations
template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<2, dealii::Vector<double>>
(const Mapping<2>                 &,
 const RMHD::Entities::FE_FieldBase<2, dealii::Vector<double>>  &,
 const dealii::Function<2>        &);
template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<3, dealii::Vector<double>>
(const Mapping<3>                 &,
 const RMHD::Entities::FE_FieldBase<3, dealii::Vector<double>>  &,
 const dealii::Function<3>        &);
template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<2, RMHD::LinearAlgebra::MPI::Vector>
(const Mapping<2>                 &,
 const RMHD::Entities::FE_FieldBase<2, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<2>        &);
template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<3, RMHD::LinearAlgebra::MPI::Vector>
(const Mapping<3>                 &,
 const RMHD::Entities::FE_FieldBase<3, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<3>        &);

template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<2, dealii::Vector<double>>
(const RMHD::Entities::FE_FieldBase<2, dealii::Vector<double>>  &,
 const dealii::Function<2>        &);
template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<3, dealii::Vector<double>>
(const RMHD::Entities::FE_FieldBase<3, dealii::Vector<double>>  &,
 const dealii::Function<3>        &);
template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<2, RMHD::LinearAlgebra::MPI::Vector>
(const RMHD::Entities::FE_FieldBase<2, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<2>        &);
template
std::map<typename dealii::VectorTools::NormType, double>
RMHD::VectorTools::compute_error<3, RMHD::LinearAlgebra::MPI::Vector>
(const RMHD::Entities::FE_FieldBase<3, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<3>        &);

template
void RMHD::VectorTools::interpolate<2, dealii::Vector<double>>
(const Mapping<2> &,
 const RMHD::Entities::FE_FieldBase<2, dealii::Vector<double>>  &,
 const dealii::Function<2>        &,
 dealii::Vector<double> &);
template
void RMHD::VectorTools::interpolate<3, dealii::Vector<double>>
(const Mapping<3>                 &,
 const RMHD::Entities::FE_FieldBase<3, dealii::Vector<double>>  &,
 const dealii::Function<3>        &,
 dealii::Vector<double>           &);
template
void RMHD::VectorTools::interpolate<2, RMHD::LinearAlgebra::MPI::Vector>
(const Mapping<2> &,
 const RMHD::Entities::FE_FieldBase<2, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<2>        &,
 RMHD::LinearAlgebra::MPI::Vector &);
template
void RMHD::VectorTools::interpolate<3, RMHD::LinearAlgebra::MPI::Vector>
(const Mapping<3>                 &,
 const RMHD::Entities::FE_FieldBase<3, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<3>        &,
 RMHD::LinearAlgebra::MPI::Vector &);


template
void RMHD::VectorTools::interpolate<2, dealii::Vector<double>>
(const RMHD::Entities::FE_FieldBase<2, dealii::Vector<double>>  &,
 const dealii::Function<2>        &,
 dealii::Vector<double>           &);
template
void RMHD::VectorTools::interpolate<3, dealii::Vector<double>>
(const RMHD::Entities::FE_FieldBase<3, dealii::Vector<double>>  &,
 const dealii::Function<3>        &,
 dealii::Vector<double>           &);
template
void RMHD::VectorTools::interpolate<2, RMHD::LinearAlgebra::MPI::Vector>
(const RMHD::Entities::FE_FieldBase<2, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<2>        &,
 RMHD::LinearAlgebra::MPI::Vector &);
template
void RMHD::VectorTools::interpolate<3, RMHD::LinearAlgebra::MPI::Vector>
(const RMHD::Entities::FE_FieldBase<3, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<3>        &,
 RMHD::LinearAlgebra::MPI::Vector &);

template
void RMHD::VectorTools::project<2, dealii::Vector<double>>
(const Mapping<2> &,
 const RMHD::Entities::FE_FieldBase<2, dealii::Vector<double>>  &,
 const dealii::Function<2>        &,
 dealii::Vector<double> &);
template
void RMHD::VectorTools::project<3, dealii::Vector<double>>
(const Mapping<3>                 &,
 const RMHD::Entities::FE_FieldBase<3, dealii::Vector<double>>  &,
 const dealii::Function<3>        &,
 dealii::Vector<double>           &);
template
void RMHD::VectorTools::project<2, RMHD::LinearAlgebra::MPI::Vector>
(const Mapping<2> &,
 const RMHD::Entities::FE_FieldBase<2, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<2>        &,
 RMHD::LinearAlgebra::MPI::Vector &);
template
void RMHD::VectorTools::project<3, RMHD::LinearAlgebra::MPI::Vector>
(const Mapping<3>                 &,
 const RMHD::Entities::FE_FieldBase<3, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<3>        &,
 RMHD::LinearAlgebra::MPI::Vector &);


template
void RMHD::VectorTools::project<2, dealii::Vector<double>>
(const RMHD::Entities::FE_FieldBase<2, dealii::Vector<double>>  &,
 const dealii::Function<2>        &,
 dealii::Vector<double>           &);
template
void RMHD::VectorTools::project<3, dealii::Vector<double>>
(const RMHD::Entities::FE_FieldBase<3, dealii::Vector<double>>  &,
 const dealii::Function<3>        &,
 dealii::Vector<double>           &);
template
void RMHD::VectorTools::project<2, RMHD::LinearAlgebra::MPI::Vector>
(const RMHD::Entities::FE_FieldBase<2, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<2>        &,
 RMHD::LinearAlgebra::MPI::Vector &);
template
void RMHD::VectorTools::project<3, RMHD::LinearAlgebra::MPI::Vector>
(const RMHD::Entities::FE_FieldBase<3, RMHD::LinearAlgebra::MPI::Vector>  &,
 const dealii::Function<3>        &,
 RMHD::LinearAlgebra::MPI::Vector &);
