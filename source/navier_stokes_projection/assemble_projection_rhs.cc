#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_projection_step_rhs()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Assembling the projection step's right hand side...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Projection step - RHS assembly");

  // Reset data
  projection_step_rhs = 0.;
  correction_step_rhs = 0.;

  // Compute the highest polynomial degree from all the integrands
  const int p_degree = velocity->fe_degree + pressure->fe_degree - 1;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<dim>    &scratch,
           AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy            &data)
    {
      this->assemble_local_projection_step_rhs(cell,
                                               scratch,
                                               data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy  &data)
    {
      this->copy_local_to_global_projection_step_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run(
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               velocity->get_dof_handler().begin_active()),
    CellFilter(IteratorFilters::LocallyOwnedCell(),
               velocity->get_dof_handler().end()),
    worker,
    copier,
    AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<dim>(
      *mapping,
      quadrature_formula,
      velocity->fe,
      update_gradients,
      pressure->fe,
      update_JxW_values |
      update_values),
    AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy(pressure->fe.dofs_per_cell));

  // Compress global data
  projection_step_rhs.compress(VectorOperation::add);
  correction_step_rhs.compress(VectorOperation::add);

  // Compute the L2 norm of the right hand side
  norm_projection_rhs = projection_step_rhs.l2_norm();

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Right-hand side's L2-norm = "
           << std::scientific << std::setprecision(6)
           << norm_projection_rhs
           << std::endl;
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_projection_step_rhs
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<dim>  &scratch,
 AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy          &data)
{
  // Reset local data
  data.local_projection_step_rhs = 0.;
  data.local_correction_step_rhs = 0.;

  // VSIMEX coefficient
  const std::vector<double> alpha = time_stepping.get_alpha();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Pressure
  scratch.pressure_fe_values.reinit(cell);

  // Velocity
  typename DoFHandler<dim>::active_cell_iterator
  velocity_cell(&velocity->get_triangulation(),
                 cell->level(),
                 cell->index(),
                &velocity->get_dof_handler());

  scratch.velocity_fe_values.reinit(velocity_cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

  scratch.velocity_fe_values[vector_extractor].get_function_divergences(
    velocity->solution,
    scratch.velocity_divergences);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      scratch.phi[i] = scratch.pressure_fe_values.shape_value(i, q);

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      const double phi_div_v =
              scratch.phi[i] *
              scratch.velocity_divergences[q] *
              scratch.pressure_fe_values.JxW(q);

      data.local_projection_step_rhs(i) -=
              alpha[0] / time_stepping.get_next_step_size() / parameters.C6 *
              phi_div_v;

      data.local_correction_step_rhs(i) -= phi_div_v;
    } // Loop over local degrees of freedom
  } // Loop over quadrature points
}

template <int dim>
void NavierStokesProjection<dim>::
copy_local_to_global_projection_step_rhs(
  const AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy   &data)
{
  phi->get_constraints().distribute_local_to_global
  (data.local_projection_step_rhs,
   data.local_dof_indices,
   projection_step_rhs);

  pressure->get_hanging_node_constraints().distribute_local_to_global
  (data.local_correction_step_rhs,
   data.local_dof_indices,
   correction_step_rhs);
}

} // namespace Step35

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_projection_step_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_projection_step_rhs();

template void RMHD::NavierStokesProjection<2>::assemble_local_projection_step_rhs
(const typename DoFHandler<2>::active_cell_iterator                         &,
 RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<2>  &,
 RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy        &);
template void RMHD::NavierStokesProjection<3>::assemble_local_projection_step_rhs
(const typename DoFHandler<3>::active_cell_iterator                         &,
 RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<3>  &,
 RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy        &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_projection_step_rhs
(const RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_projection_step_rhs
(const RMHD::AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy &);
