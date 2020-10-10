#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{
template <int dim>
void NavierStokesProjection<dim>::assemble_velocity_advection_matrix()
{
  velocity_advection_matrix = 0.;

  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AdvectionAssembly::LocalCellData<dim>                &scratch,
           AdvectionAssembly::MappingData<dim>                  &data)
    {
      this->assemble_local_velocity_advection_matrix(cell, 
                                                     scratch,
                                                     data);
    };
  
  auto copier =
    [this](const AdvectionAssembly::MappingData<dim> &data) 
    {
      this->copy_local_to_global_velocity_advection_matrix(data);
    };

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity.dof_handler.end()),
   worker,
   copier,
   AdvectionAssembly::LocalCellData<dim>(velocity.fe,
                                         velocity.quadrature_formula,
                                         update_values|
                                         update_JxW_values|
                                         update_gradients),
   AdvectionAssembly::MappingData<dim>(velocity.fe.dofs_per_cell));

  velocity_advection_matrix.compress(VectorOperation::add);

}

template <int dim>
void NavierStokesProjection<dim>::assemble_local_velocity_advection_matrix
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AdvectionAssembly::LocalCellData<dim>                 &scratch,
 AdvectionAssembly::MappingData<dim>                   &data)
{
  // reset local matrix
  data.local_matrix = 0.;

  // initialize velocity part
  const FEValuesExtractors::Vector      velocities(0);

  scratch.fe_values.reinit(cell);

  cell->get_dof_indices(data.local_dof_indices);

  scratch.fe_values[velocities].get_function_values
  (extrapolated_velocity,
   scratch.extrapolated_velocity_values);

  scratch.fe_values[velocities].get_function_divergences
  (extrapolated_velocity,
   scratch.extrapolated_velocity_divergences);

  
  if (parameters.convection_term_form == 
        RunTimeParameters::ConvectionTermForm::rotational)
    scratch.fe_values[velocities].get_function_curls(
      extrapolated_velocity, 
      scratch.extrapolated_velocity_curls);


  // loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi_velocity[i] = scratch.fe_values[velocities].value(i,q);
      scratch.grad_phi_velocity[i] = scratch.fe_values[velocities].gradient(i,q);
      scratch.curl_phi_velocity[i] = scratch.fe_values[velocities].curl(i,q);
    }
    // loop over local dofs
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        switch (parameters.convection_term_form)
        {
          case RunTimeParameters::ConvectionTermForm::standard:
          {
            data.local_matrix(i, j) += ( 
                          scratch.phi_velocity[i] *
                          scratch.grad_phi_velocity[j] *
                          scratch.extrapolated_velocity_values[q])
                          * scratch.fe_values.JxW(q);
            break;
          }
          case RunTimeParameters::ConvectionTermForm::skewsymmetric:
          {
            data.local_matrix(i, j) += ( 
                          scratch.phi_velocity[i] *
                          scratch.grad_phi_velocity[j] *
                          scratch.extrapolated_velocity_values[q]
                          +
                          0.5 *
                          scratch.extrapolated_velocity_divergences[q] *
                          scratch.phi_velocity[i] * 
                          scratch.phi_velocity[j] )
                          * scratch.fe_values.JxW(q);
            break;
          }
          case RunTimeParameters::ConvectionTermForm::divergence:
          {
            data.local_matrix(i, j) += ( 
                          scratch.phi_velocity[i] *
                          scratch.grad_phi_velocity[j] *
                          scratch.extrapolated_velocity_values[q]
                          +
                          scratch.extrapolated_velocity_divergences[q] *
                          scratch.phi_velocity[i] * 
                          scratch.phi_velocity[j] )
                          * scratch.fe_values.JxW(q);
            break;
          }
          case RunTimeParameters::ConvectionTermForm::rotational:
          {
            // This form needs to be discussed, specifically which
            // velocity instance is to be replaced by the extrapolated
            // velocity The current implementation computes the total pressure.
            // The minus sign in the argument of cross_product_2d
            // method is due to how the method is defined.
            if constexpr(dim == 2)
              data.local_matrix(i, j) += ( 
                          scratch.phi_velocity[i] *
                          scratch.curl_phi_velocity[j][0] *
                          cross_product_2d(
                            - scratch.extrapolated_velocity_values[q])) * 
                          scratch.fe_values.JxW(q);
            else if constexpr(dim == 3)
              data.local_matrix(i, j) += ( 
                          scratch.phi_velocity[i]  *
                          cross_product_3d(
                            scratch.curl_phi_velocity[j],
                            scratch.extrapolated_velocity_values[q])) *
                          scratch.fe_values.JxW(q);
            break;
          }
          default:
            Assert(false, ExcNotImplemented());
        };
    // loop over local dofs
  } // loop over quadrature points
}

template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_velocity_advection_matrix
(const AdvectionAssembly::MappingData<dim> &data)
{
  velocity.constraints.distribute_local_to_global(
                                      data.local_matrix,
                                      data.local_dof_indices,
                                      velocity_advection_matrix);
}
}

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_velocity_advection_matrix();
template void RMHD::NavierStokesProjection<3>::assemble_velocity_advection_matrix();

template void RMHD::NavierStokesProjection<2>::assemble_local_velocity_advection_matrix
(const typename DoFHandler<2>::active_cell_iterator &,
 RMHD::AdvectionAssembly::LocalCellData<2>          &,
 RMHD::AdvectionAssembly::MappingData<2>            &);
template void RMHD::NavierStokesProjection<3>::assemble_local_velocity_advection_matrix
(const typename DoFHandler<3>::active_cell_iterator &,
 RMHD::AdvectionAssembly::LocalCellData<3>          &,
 RMHD::AdvectionAssembly::MappingData<3>            &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_velocity_advection_matrix
(const RMHD::AdvectionAssembly::MappingData<2> &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_velocity_advection_matrix
(const RMHD::AdvectionAssembly::MappingData<3> &);
