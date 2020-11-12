#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_projection_step_rhs()
{
  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Projection step - RHS assembly");

  pressure_rhs = 0.;

  // Polynomial degree of the integrand

  const int p_degree = velocity->fe_degree + pressure->fe_degree - 1;

  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           PressureRightHandSideAssembly::LocalCellData<dim>    &scratch,
           PressureRightHandSideAssembly::MappingData<dim>      &data)
    {
      this->assemble_local_projection_step_rhs(cell, 
                                               scratch,
                                               data);
    };
  
  auto copier =
    [this](const PressureRightHandSideAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_projection_step_rhs(data);
    };

  WorkStream::run(CellFilter(IteratorFilters::LocallyOwnedCell(),
                             (pressure->dof_handler)->begin_active()),
                  CellFilter(IteratorFilters::LocallyOwnedCell(),
                             (pressure->dof_handler)->end()),
                  worker,
                  copier,
                  PressureRightHandSideAssembly::LocalCellData<dim>(
                                          velocity->fe,
                                          pressure->fe,
                                          quadrature_formula,
                                          update_values |
                                          update_gradients,
                                          update_JxW_values |
                                          update_values),
                  PressureRightHandSideAssembly::MappingData<dim>(
                                            pressure->fe.dofs_per_cell));
  pressure_rhs.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_projection_step_rhs
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 PressureRightHandSideAssembly::LocalCellData<dim>     &scratch,
 PressureRightHandSideAssembly::MappingData<dim>       &data)
{
  // reset local rhs
  data.local_projection_step_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  // initialize pressure part
  scratch.pressure_fe_values.reinit(cell);

  cell->get_dof_indices(data.local_pressure_dof_indices);

  // initialize velocity part
  const FEValuesExtractors::Vector  velocities(0);

  typename DoFHandler<dim>::active_cell_iterator
  velocity_cell(&velocity->get_triangulation(),
                 cell->level(),
                 cell->index(),
                velocity->dof_handler.get());

  scratch.velocity_fe_values.reinit(velocity_cell);

  scratch.velocity_fe_values[velocities].get_function_divergences(
                              velocity->solution,
                              scratch.velocity_divergence_values);

  // loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
      scratch.phi_pressure[i] = 
                          scratch.pressure_fe_values.shape_value(i, q);

    // loop over local dofs
    for (unsigned int i = 0; i < scratch.pressure_dofs_per_cell; ++i)
      data.local_projection_step_rhs(i) += 
                          - (!flag_initializing ?
                              time_stepping.get_alpha()[0] / 
                              time_stepping.get_next_step_size() :
                              1.0 / time_stepping.get_next_step_size()) *
                          scratch.pressure_fe_values.JxW(q) *
                          scratch.velocity_divergence_values[q] *
                          scratch.phi_pressure[i];
  } // loop over quadrature points
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_projection_step_rhs(
  const PressureRightHandSideAssembly::MappingData<dim>  &data)
{
  pressure->constraints.distribute_local_to_global
  (data.local_projection_step_rhs,
   data.local_pressure_dof_indices,
   pressure_rhs);
}

} // namespace Step35

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_projection_step_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_projection_step_rhs();

template void RMHD::NavierStokesProjection<2>::assemble_local_projection_step_rhs
(const typename DoFHandler<2>::active_cell_iterator     &,
 RMHD::PressureRightHandSideAssembly::LocalCellData<2>  &,
 RMHD::PressureRightHandSideAssembly::MappingData<2>    &);
template void RMHD::NavierStokesProjection<3>::assemble_local_projection_step_rhs
(const typename DoFHandler<3>::active_cell_iterator     &,
 RMHD::PressureRightHandSideAssembly::LocalCellData<3>  &,
 RMHD::PressureRightHandSideAssembly::MappingData<3>    &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_projection_step_rhs
(const RMHD::PressureRightHandSideAssembly::MappingData<2> &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_projection_step_rhs
(const RMHD::PressureRightHandSideAssembly::MappingData<3> &);
