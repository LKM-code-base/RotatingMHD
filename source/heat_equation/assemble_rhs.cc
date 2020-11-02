#include <rotatingMHD/heat_equation.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void HeatEquation<dim>::assemble_rhs()
{
  if (parameters.verbose)
    *pcout << "    Heat Equation: Assembling right hand side..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, 
                        "Heat equation: RHS assembly");

  rhs = 0.;

  // The polynomial degrees of the suply function and the neumann 
  // boundary condition function are hardcoded to match those of the 
  // temperature finite element.

  const int p_degree_supply_function = temperature.fe_degree;

  const int p_degree_neumann_function = temperature.fe_degree;

  // Maximal polynomial degree of the volume integrands

  const int p_degree = std::max(temperature.fe_degree + p_degree_supply_function,
                                2 * temperature.fe_degree + velocity->fe_degree - 1);

  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Polynomial degree of the boundary integrand

  const int face_p_degree = temperature.fe_degree + p_degree_neumann_function;

  const QGauss<dim-1>   face_quadrature_formula(std::ceil(0.5 * double(face_p_degree + 1)));

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           TemperatureRightHandSideAssembly::LocalCellData<dim>  &scratch,
           TemperatureRightHandSideAssembly::MappingData<dim>    &data)
    {
      this->assemble_local_rhs(cell, 
                                             scratch,
                                             data);
    };
  
  auto copier =
    [this](const TemperatureRightHandSideAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_rhs(data);
    };

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature.dof_handler.end()),
   worker,
   copier,
   TemperatureRightHandSideAssembly::LocalCellData<dim>(
    *mapping,
    temperature.fe,
    velocity->fe,
    quadrature_formula,
    face_quadrature_formula,
    update_values|
    update_gradients|
    update_JxW_values |
    update_quadrature_points,
    update_values |
    update_quadrature_points,
    update_JxW_values |
    update_values |
    update_quadrature_points),
   TemperatureRightHandSideAssembly::MappingData<dim>(
     temperature.fe.dofs_per_cell));

  rhs.compress(VectorOperation::add);
}

template <int dim>
void HeatEquation<dim>::assemble_local_rhs
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 TemperatureRightHandSideAssembly::LocalCellData<dim> &scratch,
 TemperatureRightHandSideAssembly::MappingData<dim>   &data)
{
  // Reset local matrix and vector
  data.local_rhs                          = 0.;
  data.local_matrix_for_inhomogeneous_bc  = 0.;

  // Prepare temperature part
  scratch.temperature_fe_values.reinit(cell);

  scratch.temperature_fe_values.get_function_values(
    temperature_tmp,
    scratch.temperature_tmp_values);

  scratch.temperature_fe_values.get_function_values(
    temperature.old_solution,
    scratch.old_temperature_values);

  scratch.temperature_fe_values.get_function_values(
    temperature.old_old_solution,
    scratch.old_old_temperature_values);

  scratch.temperature_fe_values.get_function_gradients(
    temperature.old_solution,
    scratch.old_temperature_gradients);

  scratch.temperature_fe_values.get_function_gradients(
    temperature.old_old_solution,
    scratch.old_old_temperature_gradients);

  // Prepare velocity part
  typename DoFHandler<dim>::active_cell_iterator
  velocity_cell(&temperature.dof_handler.get_triangulation(),
                 cell->level(),
                 cell->index(),
                &velocity->dof_handler);

  const FEValuesExtractors::Vector velocities(0);

  scratch.velocity_fe_values.reinit(velocity_cell);

  if (velocity != nullptr)
    scratch.velocity_fe_values[velocities].get_function_values(
      velocity->solution,
      scratch.velocity_values);
  else
    velocity_function_ptr->value_list(
      scratch.velocity_fe_values.get_quadrature_points(),
      scratch.velocity_values);
  
  // Supply term
  if (supply_term_ptr != nullptr)
    supply_term_ptr->value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.supply_term_values);
  else
    ZeroFunction<dim>().value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.supply_term_values);

  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Populate test functions
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.temperature_fe_values.shape_value(i, q);
      scratch.grad_phi[i] = scratch.temperature_fe_values.shape_grad(i, q);
    }

    // Loop over local DoFs
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=  scratch.temperature_fe_values.JxW(q) *
                            (scratch.phi[i]*
                             scratch.supply_term_values[q]
                             -
                             scratch.phi[i] *
                             scratch.temperature_tmp_values[q]
                             -
                             time_stepping.get_gamma()[1] /
                             parameters.Re /
                             parameters.Pr *
                             scratch.grad_phi[i] *
                             scratch.old_temperature_gradients[q]
                             -
                             time_stepping.get_gamma()[2] /
                             parameters.Re /
                             parameters.Pr *
                             scratch.grad_phi[i] *
                             scratch.old_old_temperature_gradients[q]);
      if (!parameters.flag_semi_implicit_convection &&
          !flag_ignore_advection)
        data.local_rhs(i) -= scratch.temperature_fe_values.JxW(q) *
                             (time_stepping.get_beta()[0] *
                              scratch.phi[i] *
                              scratch.velocity_values[q] *
                              scratch.old_temperature_gradients[q]
                              +
                              time_stepping.get_beta()[1] *
                              scratch.phi[i] *
                              scratch.velocity_values[q] *
                              scratch.old_old_temperature_gradients[q]);
      if (temperature.constraints.is_inhomogeneously_constrained(
          data.local_dof_indices[i]))
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        {
          data.local_matrix_for_inhomogeneous_bc(j, i) += 
                scratch.temperature_fe_values.JxW(q) * (
                time_stepping.get_alpha()[0] /
                time_stepping.get_next_step_size() *
                scratch.phi[j] *
                scratch.phi[i]
                +
                time_stepping.get_gamma()[0] /
                parameters.Re /
                parameters.Pr *
                scratch.grad_phi[j] *
                scratch.grad_phi[i]);
          if (parameters.flag_semi_implicit_convection &&
              !flag_ignore_advection)
            data.local_matrix_for_inhomogeneous_bc(j, i) +=
                scratch.temperature_fe_values.JxW(q) *
                scratch.phi[j] *
                scratch.velocity_values[q] *
                scratch.grad_phi[i];
        }
    }
  }
  for (const auto &face : cell->face_iterators())
    if (face->at_boundary() && 
        temperature.boundary_conditions.neumann_bcs.find(face->boundary_id()) 
          != temperature.boundary_conditions.neumann_bcs.end())
    {
      scratch.temperatuer_fe_face_values.reinit(cell, face);

      temperature.boundary_conditions.neumann_bcs[face->boundary_id()]->value_list(
        scratch.temperatuer_fe_face_values.get_quadrature_points(),
        scratch.supply_term_values);

      for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
      {
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          scratch.face_phi[i] = 
                    scratch.temperatuer_fe_face_values.shape_value(i, q);
        
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          data.local_rhs(i) += scratch.temperatuer_fe_face_values.JxW(q) *
                              scratch.face_phi[i] *
                              scratch.supply_term_values[q];
      }
    }
}

template <int dim>
void HeatEquation<dim>::copy_local_to_global_rhs
(const TemperatureRightHandSideAssembly::MappingData<dim> &data)
{
  temperature.constraints.distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    rhs,
    data.local_matrix_for_inhomogeneous_bc);
}


} // namespace RMHD

// explicit instantiations
template void RMHD::HeatEquation<2>::assemble_rhs();
template void RMHD::HeatEquation<3>::assemble_rhs();

template void RMHD::HeatEquation<2>::assemble_local_rhs
(const typename DoFHandler<2>::active_cell_iterator  &,
 RMHD::TemperatureRightHandSideAssembly::LocalCellData<2>    &,
 RMHD::TemperatureRightHandSideAssembly::MappingData<2>      &);
template void RMHD::HeatEquation<3>::assemble_local_rhs
(const typename DoFHandler<3>::active_cell_iterator  &,
 RMHD::TemperatureRightHandSideAssembly::LocalCellData<3>    &,
 RMHD::TemperatureRightHandSideAssembly::MappingData<3>      &);

template void RMHD::HeatEquation<2>::copy_local_to_global_rhs
(const RMHD::TemperatureRightHandSideAssembly::MappingData<2>  &);
template void RMHD::HeatEquation<3>::copy_local_to_global_rhs
(const RMHD::TemperatureRightHandSideAssembly::MappingData<3>  &);