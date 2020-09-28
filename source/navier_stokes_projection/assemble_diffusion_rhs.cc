#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step_rhs()
{
  velocity_rhs  = 0.;

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           VelocityRightHandSideAssembly::LocalCellData<dim>    &scratch,
           VelocityRightHandSideAssembly::MappingData<dim>      &data)
    {
      this->assemble_local_diffusion_step_rhs(cell, 
                                              scratch,
                                              data);
    };
  
  auto copier =
    [this](const VelocityRightHandSideAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_diffusion_step_rhs(data);
    };

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.end()),
   worker,
   copier,
   VelocityRightHandSideAssembly::LocalCellData<dim>(velocity.fe,
                                                     pressure.fe,
                                                     velocity.quadrature_formula,
                                                     update_values|
                                                     update_gradients|
                                                     update_JxW_values,
                                                     update_values),
   VelocityRightHandSideAssembly::MappingData<dim>(velocity.fe.dofs_per_cell));

  velocity_rhs.compress(VectorOperation::add);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 VelocityRightHandSideAssembly::LocalCellData<dim>     &scratch,
 VelocityRightHandSideAssembly::MappingData<dim>       &data)
{
  // reset local matrix and vector
  data.local_diffusion_step_rhs = 0.;
  data.local_matrix_for_inhomogeneous_bc = 0.;

  // prepare velocity part
  const FEValuesExtractors::Vector  velocities(0);

  scratch.velocity_fe_values.reinit(cell);
  
  cell->get_dof_indices(data.local_velocity_dof_indices);

  scratch.velocity_fe_values[velocities].get_function_values
  (velocity_tmp,
   scratch.velocity_tmp_values);

  // prepare pressure part
  typename DoFHandler<dim>::active_cell_iterator
  pressure_cell(&velocity.dof_handler.get_triangulation(),
                 cell->level(),
                 cell->index(),
                &pressure.dof_handler);

  scratch.pressure_fe_values.reinit(pressure_cell);
  
  scratch.pressure_fe_values.get_function_values
  (pressure_tmp,
  scratch.pressure_tmp_values);

  // The velocity gradients of previous solutions are
  // needed for the laplacian term
  scratch.velocity_fe_values[velocities].get_function_gradients
  (velocity.old_solution,
   scratch.old_velocity_gradients);
  
  scratch.velocity_fe_values[velocities].get_function_gradients
  (velocity.old_old_solution,
   scratch.old_old_velocity_gradients);

  // The velocity and its divergence of previous solutions are
  // needed for the convection term
  if (!parameters.flag_semi_implicit_scheme)
  {
    scratch.velocity_fe_values[velocities].get_function_values
    (velocity.old_solution,
    scratch.old_velocity_values);
    scratch.velocity_fe_values[velocities].get_function_divergences
    (velocity.old_solution,
    scratch.old_velocity_divergences);
    scratch.velocity_fe_values[velocities].get_function_curls
    (velocity.old_solution,
    scratch.old_velocity_curls);
    
    scratch.velocity_fe_values[velocities].get_function_values
    (velocity.old_old_solution,
    scratch.old_old_velocity_values);
    scratch.velocity_fe_values[velocities].get_function_divergences
    (velocity.old_old_solution,
    scratch.old_old_velocity_divergences);
    scratch.velocity_fe_values[velocities].get_function_curls
    (velocity.old_old_solution,
    scratch.old_old_velocity_curls);
  }

  // loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      scratch.phi_velocity[i] = 
                  scratch.velocity_fe_values[velocities].value(i, q);
      scratch.div_phi_velocity[i] =
                  scratch.velocity_fe_values[velocities].divergence(i, q);
      scratch.grad_phi_velocity[i] = 
                scratch.velocity_fe_values[velocities].gradient(i, q);
    }
    
    // loop over local dofs
    for (unsigned int i = 0; i < scratch.velocity_dofs_per_cell; ++i)
    {
      data.local_diffusion_step_rhs(i) +=
                                  scratch.velocity_fe_values.JxW(q) * (
                                  scratch.pressure_tmp_values[q] *
                                  scratch.div_phi_velocity[i]
                                  - 
                                  scratch.velocity_tmp_values[q] *
                                  scratch.phi_velocity[i]);
      if (parameters.flag_full_vsimex_scheme)
        {
          data.local_diffusion_step_rhs(i) -=
                                    scratch.velocity_fe_values.JxW(q) * (
                                    time_stepping.get_gamma()[1] /
                                    parameters.Re *
                                    scalar_product(
                                    scratch.old_velocity_gradients[q],
                                    scratch.grad_phi_velocity[i])
                                    +
                                    time_stepping.get_gamma()[2] /
                                    parameters.Re *
                                    scalar_product(
                                    scratch.old_old_velocity_gradients[q],
                                    scratch.grad_phi_velocity[i]));
          if (!parameters.flag_semi_implicit_scheme)
            switch (parameters.convection_term_form)
            {
              case RunTimeParameters::ConvectionTermForm::standard:
              {
                data.local_diffusion_step_rhs(i) -=
                      scratch.velocity_fe_values.JxW(q) * (
                      time_stepping.get_beta()[0] *
                      scratch.phi_velocity[i] *
                      scratch.old_velocity_gradients[q] *  
                      scratch.old_velocity_values[q]
                      +
                      time_stepping.get_beta()[1] *
                      scratch.phi_velocity[i] *
                      scratch.old_old_velocity_gradients[q] *  
                      scratch.old_old_velocity_values[q]);
                break;
              }
              case RunTimeParameters::ConvectionTermForm::skewsymmetric:
              {
                data.local_diffusion_step_rhs(i) -=
                      scratch.velocity_fe_values.JxW(q) * (
                      time_stepping.get_beta()[0] *
                      (scratch.phi_velocity[i] *
                      scratch.old_velocity_gradients[q] *
                      scratch.old_velocity_values[q]
                      +
                      0.5 *
                      scratch.old_velocity_divergences[q] *
                      scratch.old_velocity_values[q] * 
                      scratch.phi_velocity[i])
                      +
                      time_stepping.get_beta()[1] *
                      (scratch.phi_velocity[i] *
                      scratch.old_old_velocity_gradients[q] *
                      scratch.old_old_velocity_values[q]
                      +
                      0.5 *
                      scratch.old_old_velocity_divergences[q] *
                      scratch.old_old_velocity_values[q] * 
                      scratch.phi_velocity[i]));
                break;
              }
              case RunTimeParameters::ConvectionTermForm::divergence:
              {
                data.local_diffusion_step_rhs(i) -=
                      scratch.velocity_fe_values.JxW(q) * (
                      time_stepping.get_beta()[0] *
                      (scratch.phi_velocity[i] *
                      scratch.old_velocity_gradients[q] *
                      scratch.old_velocity_values[q]
                      +
                      scratch.old_velocity_divergences[q] *
                      scratch.old_velocity_values[q] *
                      scratch.phi_velocity[i])
                      +
                      time_stepping.get_beta()[1] *
                      (scratch.phi_velocity[i] *
                      scratch.old_old_velocity_gradients[q] *
                      scratch.old_old_velocity_values[q]
                      +
                      scratch.old_old_velocity_divergences[q] *
                      scratch.old_old_velocity_values[q] * 
                      scratch.phi_velocity[i]));
                break;
              }
              case RunTimeParameters::ConvectionTermForm::rotational:
              {
                // The minus sign in the argument of cross_product_2d
                // method is due to how the method is defined.
                if constexpr(dim == 2)
                  data.local_diffusion_step_rhs(i) -=
                        scratch.velocity_fe_values.JxW(q) * (
                        time_stepping.get_beta()[1] *
                        (scratch.phi_velocity[i] *
                        scratch.old_velocity_curls[q][0] *
                        cross_product_2d(
                          - scratch.old_velocity_values[q]))
                        +
                        time_stepping.get_beta()[2] *
                        (scratch.phi_velocity[i] *
                        scratch.old_old_velocity_curls[q][0] *
                        cross_product_2d(
                          - scratch.old_old_velocity_values[q])));
                else if constexpr(dim == 3)
                  data.local_diffusion_step_rhs(i) -=
                        scratch.velocity_fe_values.JxW(q) * (
                        time_stepping.get_beta()[1] *
                        (scratch.phi_velocity[i] *
                        cross_product_3d(
                          scratch.old_velocity_curls[q],
                          scratch.old_velocity_values[q]))
                        +
                        time_stepping.get_beta()[2] *
                        (scratch.phi_velocity[i]  *
                        cross_product_3d(
                          scratch.old_old_velocity_curls[q],
                          scratch.old_old_velocity_values[q])));
                break;
              }
              default:
                Assert(false, ExcNotImplemented());
            };
        }

      // assemble matrix for inhomogeneous boundary conditions
      if (velocity.constraints.is_inhomogeneously_constrained(
          data.local_velocity_dof_indices[i]))
      {
        // The values inside the scope are only needed for the
        // semi-implicit scheme.
        if (parameters.flag_semi_implicit_scheme || flag_initializing)
        {
          scratch.velocity_fe_values[velocities].get_function_values(
                                    extrapolated_velocity, 
                                    scratch.extrapolated_velocity_values);
          scratch.velocity_fe_values[velocities].get_function_divergences(
                              extrapolated_velocity, 
                              scratch.extrapolated_velocity_divergences);
          scratch.velocity_fe_values[velocities].get_function_curls(
                              extrapolated_velocity, 
                              scratch.extrapolated_velocity_curls);
        }

        for (unsigned int j = 0; j < scratch.velocity_dofs_per_cell; ++j)
        {
          /* The following if scope is added to reuse the assembly
          method for the diffusion prestep */ 
          if (!flag_initializing)
          {
            data.local_matrix_for_inhomogeneous_bc(j, i) += (
                              time_stepping.get_alpha()[0] / 
                              time_stepping.get_next_step_size() *
                              scratch.phi_velocity[j] *
                              scratch.phi_velocity[i]
                              +
                              ((parameters.flag_full_vsimex_scheme) ?
                                time_stepping.get_gamma()[0] :
                                1.0) /
                              parameters.Re *
                              scalar_product(
                                scratch.grad_phi_velocity[j],
                                scratch.grad_phi_velocity[i]))  
                              * scratch.velocity_fe_values.JxW(q);
          }
          else
          {
            data.local_matrix_for_inhomogeneous_bc(j, i) += (
                              1.0 / time_stepping.get_next_step_size() *
                              scratch.phi_velocity[j] *
                              scratch.phi_velocity[i]
                              +
                              1.0 / parameters.Re *
                              scalar_product(
                                scratch.grad_phi_velocity[j],
                                scratch.grad_phi_velocity[i]))  
                              * scratch.velocity_fe_values.JxW(q);
          }
          if (parameters.flag_semi_implicit_scheme || flag_initializing)
            switch (parameters.convection_term_form)
            {
              case RunTimeParameters::ConvectionTermForm::standard:
              {
                data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      scratch.velocity_fe_values.JxW(q) * (
                      scratch.phi_velocity[j] *
                      scratch.grad_phi_velocity[i] *
                      scratch.extrapolated_velocity_values[q]);
                break;
              }
              case RunTimeParameters::ConvectionTermForm::skewsymmetric:
              {
                data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      scratch.velocity_fe_values.JxW(q) * (
                      scratch.phi_velocity[j] *
                      scratch.grad_phi_velocity[i] *  
                      scratch.extrapolated_velocity_values[q]            
                      +                                    
                      0.5 *                                
                      scratch.extrapolated_velocity_divergences[q] *            
                      scratch.phi_velocity[j] * 
                      scratch.phi_velocity[i]);
                break;
              }
              case RunTimeParameters::ConvectionTermForm::divergence:
              {
                data.local_matrix_for_inhomogeneous_bc(j, i) +=
                      scratch.velocity_fe_values.JxW(q) * (
                      scratch.phi_velocity[j] *
                      scratch.grad_phi_velocity[i] *  
                      scratch.extrapolated_velocity_values[q]
                      +
                      scratch.extrapolated_velocity_divergences[q] *
                      scratch.phi_velocity[j] * 
                      scratch.phi_velocity[i]);
                break;
              }
              case RunTimeParameters::ConvectionTermForm::rotational:
              {
                // This form needs to be discussed, specifically which
                // velocity instance is to be replaced by the extrapolated
                // velocity.
                // The minus sign in the argument of cross_product_2d
                // method is due to how the method is defined.
                if constexpr(dim == 2)
                  data.local_matrix_for_inhomogeneous_bc(j, i) +=
                        scratch.velocity_fe_values.JxW(q) * (
                        scratch.phi_velocity[j] *
                        scratch.extrapolated_velocity_curls[q][0] *
                        cross_product_2d(
                          - scratch.phi_velocity[i]));
                else if constexpr(dim == 3)
                  data.local_matrix_for_inhomogeneous_bc(j, i) +=
                        scratch.velocity_fe_values.JxW(q) * (
                        scratch.phi_velocity[j] *
                        cross_product_3d(
                          scratch.extrapolated_velocity_curls[q],
                          - scratch.phi_velocity[i]));
                break;
              }
              default:
                Assert(false, ExcNotImplemented());
            };
        }
      }
    } // loop over local dofs
  } // loop over quadrature points
}

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_diffusion_step_rhs
(const VelocityRightHandSideAssembly::MappingData<dim>  &data)
{
  velocity.constraints.distribute_local_to_global(
                                data.local_diffusion_step_rhs,
                                data.local_velocity_dof_indices,
                                velocity_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step_rhs();

template void RMHD::NavierStokesProjection<2>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<2>::active_cell_iterator     &,
 RMHD::VelocityRightHandSideAssembly::LocalCellData<2>  &,
 RMHD::VelocityRightHandSideAssembly::MappingData<2>    &);
template void RMHD::NavierStokesProjection<3>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<3>::active_cell_iterator     &,
 RMHD::VelocityRightHandSideAssembly::LocalCellData<3>  &,
 RMHD::VelocityRightHandSideAssembly::MappingData<3>    &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_diffusion_step_rhs
(const RMHD::VelocityRightHandSideAssembly::MappingData<2>  &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_diffusion_step_rhs
(const RMHD::VelocityRightHandSideAssembly::MappingData<3>  &);
