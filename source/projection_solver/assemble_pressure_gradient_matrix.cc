#include <rotatingMHD/projection_solver.h>
#include <deal.II/base/work_stream.h>

namespace Step35
{
  template <int dim>
  void NavierStokesProjection<dim>::
  assemble_p_gradient_matrix()
  {
    PressureGradientTermAssembly::MappingData<dim> per_task_data(
                                    v_fe.dofs_per_cell,
                                    p_fe.dofs_per_cell);
    PressureGradientTermAssembly::LocalCellData<dim> scratch_data(
                                    v_fe,
                                    p_fe,
                                    v_quadrature_formula,
                                    update_gradients | 
                                    update_JxW_values,
                                    update_values);

    WorkStream::run(
      IteratorPair(IteratorTuple(v_dof_handler.begin_active(),
                                 p_dof_handler.begin_active())),
      IteratorPair(IteratorTuple(v_dof_handler.end(),
                                 p_dof_handler.end())),
      *this,
      &NavierStokesProjection<dim>::local_assemble_p_gradient_matrix,
      &NavierStokesProjection<dim>::mapping_p_gradient_matrix,
      scratch_data,
      per_task_data);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  local_assemble_p_gradient_matrix(
    const IteratorPair & SI,
    PressureGradientTermAssembly::LocalCellData<dim>  &scratch,
    PressureGradientTermAssembly::MappingData<dim>    &data)
  {
    scratch.v_fe_values.reinit(std::get<0>(*SI));
    scratch.p_fe_values.reinit(std::get<1>(*SI));

    std::get<0>(*SI)->get_dof_indices(data.v_local_dof_indices);
    std::get<1>(*SI)->get_dof_indices(data.p_local_dof_indices);

    const FEValuesExtractors::Vector  velocity(0);

    data.local_matrix = 0.;
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    {
      for (unsigned int i = 0; i < data.v_dofs_per_cell; ++i)
        scratch.div_phi_v[i] = 
                        scratch.v_fe_values[velocity].divergence(i, q);
      for (unsigned int i = 0; i < data.p_dofs_per_cell; ++i)
        scratch.phi_p[i] = scratch.p_fe_values.shape_value(i,q);

      for (unsigned int i = 0; i < data.v_dofs_per_cell; ++i)
        for (unsigned int j = 0; j < data.p_dofs_per_cell; ++j)
          data.local_matrix(i, j) += -scratch.v_fe_values.JxW(q) *
                                      scratch.phi_p[j] *
                                      scratch.div_phi_v[i];
    }
  }


  template <int dim>
  void NavierStokesProjection<dim>::
  mapping_p_gradient_matrix(
    const PressureGradientTermAssembly::MappingData<dim> &data)
  {
    for (unsigned int i = 0; i < data.v_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < data.p_dofs_per_cell; ++j)
        p_gradient_matrix.add(data.v_local_dof_indices[i],
                              data.p_local_dof_indices[j],
                              data.local_matrix(i, j));
  }
}

template void Step35::NavierStokesProjection<2>::assemble_p_gradient_matrix();
template void Step35::NavierStokesProjection<3>::assemble_p_gradient_matrix();
template void Step35::NavierStokesProjection<2>::local_assemble_p_gradient_matrix(
    const IteratorPair &,
    Step35::PressureGradientTermAssembly::LocalCellData<2>  &,
    Step35::PressureGradientTermAssembly::MappingData<2>    &);
template void Step35::NavierStokesProjection<3>::local_assemble_p_gradient_matrix(
    const IteratorPair &, 
    Step35::PressureGradientTermAssembly::LocalCellData<3>  &,
    Step35::PressureGradientTermAssembly::MappingData<3>    &);
template void Step35::NavierStokesProjection<2>::mapping_p_gradient_matrix(
    const Step35::PressureGradientTermAssembly::MappingData<2> &);
template void Step35::NavierStokesProjection<3>::mapping_p_gradient_matrix(
    const Step35::PressureGradientTermAssembly::MappingData<3> &);