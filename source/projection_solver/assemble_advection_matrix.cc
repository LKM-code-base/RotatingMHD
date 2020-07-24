#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>

namespace Step35
{
  template <int dim>
  void NavierStokesProjection<dim>::assemble_v_advection_matrix()
  {
    v_advection_matrix = 0.;
    AdvectionTermAssembly::MappingData<dim> data(
                                v_fe.dofs_per_cell);
    AdvectionTermAssembly::LocalCellData<dim> scratch(
                                v_fe,
                                v_quadrature_formula,
                                update_values | 
                                update_JxW_values |
                                update_gradients);
    WorkStream::run(
      v_dof_handler.begin_active(),
      v_dof_handler.end(),
      *this,
      &NavierStokesProjection<dim>::local_assemble_v_advection_matrix,
      &NavierStokesProjection<dim>::mapping_v_advection_matrix,
      scratch,
      data);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  local_assemble_v_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    AdvectionTermAssembly::LocalCellData<dim>             &scratch,
    AdvectionTermAssembly::MappingData<dim>               &data)
  {
    scratch.fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
    const FEValuesExtractors::Vector      velocity(0);

    scratch.fe_values[velocity].get_function_values(
                                        v_extrapolated, 
                                        scratch.v_extrapolated_values);
    scratch.fe_values[velocity].get_function_divergences(
                                    v_extrapolated, 
                                    scratch.v_extrapolated_divergence);

    data.local_matrix = 0.;
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    {
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      {
        scratch.phi_v[i] = scratch.fe_values[velocity].value(i,q);
        scratch.grad_phi_v[i] = scratch.fe_values[velocity].gradient(i,q);
      }
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          data.local_matrix(i, j) += ( 
                              scratch.phi_v[i] *
                              scratch.grad_phi_v[j] *  
                              scratch.v_extrapolated_values[q]            
                              +                                    
                              0.5 *                                
                              scratch.v_extrapolated_divergence[q] *            
                              scratch.phi_v[i] * 
                              scratch.phi_v[j])  
                              * scratch.fe_values.JxW(q);
    }
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  mapping_v_advection_matrix(
    const AdvectionTermAssembly::MappingData<dim> &data)
  {
    for (unsigned int i = 0; i < v_fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < v_fe.dofs_per_cell; ++j)
        v_advection_matrix.add(data.local_dof_indices[i],
                          data.local_dof_indices[j],
                          data.local_matrix(i, j));
  }
}

template void Step35::NavierStokesProjection<2>::assemble_v_advection_matrix();
template void Step35::NavierStokesProjection<3>::assemble_v_advection_matrix();
template void Step35::NavierStokesProjection<2>::local_assemble_v_advection_matrix(
    const typename DoFHandler<2>::active_cell_iterator  &,
    Step35::AdvectionTermAssembly::LocalCellData<2>             &,
    Step35::AdvectionTermAssembly::MappingData<2>               &);
template void Step35::NavierStokesProjection<3>::local_assemble_v_advection_matrix(
    const typename DoFHandler<3>::active_cell_iterator  &,
    Step35::AdvectionTermAssembly::LocalCellData<3>             &,
    Step35::AdvectionTermAssembly::MappingData<3>               &);
template void Step35::NavierStokesProjection<2>::mapping_v_advection_matrix(
    const Step35::AdvectionTermAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::mapping_v_advection_matrix(
    const Step35::AdvectionTermAssembly::MappingData<3> &);