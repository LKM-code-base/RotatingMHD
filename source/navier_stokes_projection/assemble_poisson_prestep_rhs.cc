#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_poisson_prestep_rhs()
{
  poisson_prestep_rhs = 0.;

  const FE_Q<dim> dummy_fe(1);

  const FE_Q<dim> * const temperature_fe_ptr = 
          (temperature.get() != nullptr) ? &temperature->fe : &dummy_fe;

  // Polynomial degree of the body force

  const int p_degree_body_force = velocity->fe_degree;

  // Polynomial degree of the integrand

  const int p_degree = p_degree_body_force + pressure->fe_degree - 1;

  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Polynomial degree of the boundary integrand

  const int face_p_degree = std::max(pressure->fe_degree + p_degree_body_force,
                                     pressure->fe_degree + velocity->fe_degree -2);

  const QGauss<dim-1>   face_quadrature_formula(std::ceil(0.5 * double(face_p_degree + 1)));

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           PoissonPrestepRightHandSideAssembly::LocalCellData<dim>  &scratch,
           PoissonPrestepRightHandSideAssembly::MappingData<dim>    &data)
    {
      this->assemble_local_poisson_prestep_rhs(cell, 
                                               scratch,
                                               data);
    };
  
  auto copier =
    [this](const PoissonPrestepRightHandSideAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_poisson_prestep_rhs(data);
    };

  WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                             (pressure->dof_handler)->begin_active()),
                  CellFilter(IteratorFilters::LocallyOwnedCell(),
                             (pressure->dof_handler)->end()),
                  worker,
                  copier,
                  PoissonPrestepRightHandSideAssembly::LocalCellData<dim>(
                                  velocity->fe,
                                  pressure->fe,
                                  *temperature_fe_ptr,
                                  quadrature_formula,
                                  face_quadrature_formula,
                                  update_hessians,
                                  update_values|
                                  update_gradients|
                                  update_JxW_values|
                                  update_quadrature_points,
                                  update_values|
                                  update_JxW_values|
                                  update_normal_vectors|
                                  update_quadrature_points,
                                  update_gradients,
                                  update_values),
                  PoissonPrestepRightHandSideAssembly::MappingData<dim>(
                                          pressure->fe.dofs_per_cell));
  poisson_prestep_rhs.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_poisson_prestep_rhs
(const typename DoFHandler<dim>::active_cell_iterator        &cell,
 PoissonPrestepRightHandSideAssembly::LocalCellData<dim>     &scratch,
 PoissonPrestepRightHandSideAssembly::MappingData<dim>       &data)
{
  data.local_poisson_prestep_rhs = 0.;

  data.local_matrix_for_inhomogeneous_bc = 0.;

  scratch.pressure_fe_values.reinit(cell);

  // Prepare temperature part
  if (!flag_ignore_temperature)
  {
    typename DoFHandler<dim>::active_cell_iterator
    temperature_cell(&pressure->get_triangulation(),
                     cell->level(),
                     cell->index(),
                     (temperature->dof_handler).get());

    scratch.temperature_fe_values.reinit(temperature_cell);

    scratch.temperature_fe_values.get_function_gradients(
      temperature->old_old_solution,
      scratch.temperature_gradient_values);
  }

  cell->get_dof_indices(data.local_pressure_dof_indices);
  
  const FEValuesExtractors::Vector  velocities(0);

  if (body_force_ptr != nullptr)
    body_force_ptr->divergence_list(
      scratch.pressure_fe_values.get_quadrature_points(),
      scratch.body_force_divergence_values);
  else
    scratch.body_force_divergence_values = 
                      std::vector<double>(scratch.n_q_points, 0.0);

  /*! @attention Quick fix for the benchmark. A more general approach
      has to be come up with */
  Tensor<1, dim>  downwards_unit_vector;
  downwards_unit_vector[dim-1] = 1.0;

  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
      scratch.phi_pressure[i] = 
                          scratch.pressure_fe_values.shape_value(i, q);

    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
    {
      data.local_poisson_prestep_rhs(i) -= 
                          scratch.pressure_fe_values.JxW(q) *
                          scratch.body_force_divergence_values[q] *
                          scratch.phi_pressure[i];
      
      if (!flag_ignore_temperature)
        data.local_poisson_prestep_rhs(i) -=
                          scratch.pressure_fe_values.JxW(q) *
                          scratch.temperature_gradient_values[q] *
                          downwards_unit_vector *
                          scratch.phi_pressure[i];

      if (pressure->constraints.is_inhomogeneously_constrained(
        data.local_pressure_dof_indices[i]))
      {
        for (unsigned int k = 0; k < scratch.pressure_dofs_per_cell; ++k)
          scratch.grad_phi_pressure[k] =
                          scratch.pressure_fe_values.shape_grad(k, q);

        for (unsigned int j = 0; j < scratch.pressure_dofs_per_cell; ++j)
          data.local_matrix_for_inhomogeneous_bc(j, i) +=
                                    scratch.pressure_fe_values.JxW(q) *
                                    scratch.grad_phi_pressure[i] *
                                    scratch.grad_phi_pressure[j];
      }
    } 
  }
  /*!
   * @todo Formulation: Is this correct if there are other types of 
   * boundary conditions apart from Dirichlet?
   */
  for (const auto &face : cell->face_iterators())
    if (face->at_boundary())
    {
      scratch.pressure_fe_face_values.reinit(cell, face);

      // Prepare velocity part
      typename DoFHandler<dim>::active_cell_iterator
      velocity_cell(&velocity->get_triangulation(),
                     cell->level(),
                     cell->index(),
                    velocity->dof_handler.get());

      typename DoFHandler<dim>::active_face_iterator
      velocity_face(&velocity->get_triangulation(),
                     face->level(),
                     face->index(),
                    velocity->dof_handler.get());

      scratch.velocity_fe_face_values.reinit(velocity_cell, velocity_face);

      scratch.velocity_fe_face_values[velocities].get_function_laplacians(
                                    velocity->old_old_solution,
                                    scratch.velocity_laplacian_values);
      // Prepare temperature part
      if (!flag_ignore_temperature)
      {
        typename DoFHandler<dim>::active_cell_iterator
        temperature_cell(&pressure->get_triangulation(),
                         cell->level(),
                         cell->index(),
                         (temperature->dof_handler).get());

        typename DoFHandler<dim>::active_face_iterator
        temperature_face(&velocity->get_triangulation(),
                         face->level(),
                         face->index(),
                         temperature->dof_handler.get());

        scratch.temperature_fe_face_values.reinit(temperature_cell,
                                                  temperature_face);
        scratch.temperature_fe_face_values.get_function_values(
          temperature->old_old_solution,
          scratch.temperature_values);
      }

      if (body_force_ptr != nullptr)
        body_force_ptr->value_list(
          scratch.pressure_fe_face_values.get_quadrature_points(),
          scratch.body_force_values);
      else
        scratch.body_force_values = 
                          std::vector<Tensor<1,dim>>(scratch.n_face_q_points,
                                                     Tensor<1,dim>());

      scratch.normal_vectors = 
                    scratch.pressure_fe_face_values.get_normal_vectors();

      for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
        {
          for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
            scratch.face_phi_pressure[i] = 
                        scratch.pressure_fe_face_values.shape_value(i, q);

          for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
          {
            data.local_poisson_prestep_rhs(i) += 
                            scratch.face_phi_pressure[i] * 
                            (scratch.body_force_values[q]
                             +
                             1.0 / parameters.Re *
                             scratch.velocity_laplacian_values[q])*
                            scratch.normal_vectors[q] *
                            scratch.pressure_fe_face_values.JxW(q);  
            if (!flag_ignore_temperature)
              data.local_poisson_prestep_rhs(i) +=
                            scratch.pressure_fe_face_values.JxW(q) *
                            scratch.face_phi_pressure[i] *
                            scratch.temperature_values[q] *
                            downwards_unit_vector *
                            scratch.normal_vectors[q];
          }
        }
    }
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_poisson_prestep_rhs(
  const PoissonPrestepRightHandSideAssembly::MappingData<dim>  &data)
{
  pressure->constraints.distribute_local_to_global(
                                data.local_poisson_prestep_rhs,
                                data.local_pressure_dof_indices,
                                poisson_prestep_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

template void RMHD::NavierStokesProjection<2>::assemble_poisson_prestep_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_poisson_prestep_rhs();
template void RMHD::NavierStokesProjection<2>::assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<2>::active_cell_iterator              &,
    RMHD::PoissonPrestepRightHandSideAssembly::LocalCellData<2>     &,
    RMHD::PoissonPrestepRightHandSideAssembly::MappingData<2>       &);
template void RMHD::NavierStokesProjection<3>::assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<3>::active_cell_iterator              &,
    RMHD::PoissonPrestepRightHandSideAssembly::LocalCellData<3>     &,
    RMHD::PoissonPrestepRightHandSideAssembly::MappingData<3>       &);
template void RMHD::NavierStokesProjection<2>::copy_local_to_global_poisson_prestep_rhs(
    const RMHD::PoissonPrestepRightHandSideAssembly::MappingData<2> &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_poisson_prestep_rhs(
    const RMHD::PoissonPrestepRightHandSideAssembly::MappingData<3> &);
