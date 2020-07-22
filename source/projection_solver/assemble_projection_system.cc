/*
 * assemble_projection_system.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/base/work_stream.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>

#include <fstream>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::initialize_gradient_operator()
{
  {
    DynamicSparsityPattern dsp(dof_handler_velocity.n_dofs(),
                               dof_handler_pressure.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler_velocity,
                                    dof_handler_pressure,
                                    dsp);
    sparsity_pattern_pres_vel.copy_from(dsp);
    std::ofstream out("mixed_sparsity_pattern.gpl");
    sparsity_pattern_pres_vel.print_gnuplot(out);
  }

  InitGradPerTaskData per_task_data(fe_velocity.dofs_per_cell,
                                    fe_pressure.dofs_per_cell);
  InitGradScratchData scratch_data(fe_velocity,
                                   fe_pressure,
                                   quadrature_velocity,
                                   update_gradients | update_JxW_values,
                                   update_values);

  //for (unsigned int d = 0; d < dim; ++d)
  //  {
  pres_Diff.reinit(sparsity_pattern_pres_vel);
  WorkStream::run(IteratorPair(IteratorTuple(dof_handler_velocity.begin_active(),
                                             dof_handler_pressure.begin_active())),
                  IteratorPair(IteratorTuple(dof_handler_velocity.end(),
                                             dof_handler_pressure.end())),
                 *this,
                 &NavierStokesProjection<dim>::assemble_one_cell_of_gradient,
                 &NavierStokesProjection<dim>::copy_gradient_local_to_global,
                  scratch_data,
                  per_task_data);
  //  }
}

template <int dim>
void NavierStokesProjection<dim>::assemble_one_cell_of_gradient
(const IteratorPair & SI,
 InitGradScratchData &scratch,
 InitGradPerTaskData &data)
{
  scratch.fe_val_vel.reinit(std::get<0>(*SI));
  scratch.fe_val_pres.reinit(std::get<1>(*SI));

  std::get<0>(*SI)->get_dof_indices(data.vel_local_dof_indices);
  std::get<1>(*SI)->get_dof_indices(data.pres_local_dof_indices);

  const FEValuesExtractors::Vector  velocity(0);

  data.local_grad = 0.;
  for (unsigned int q = 0; q < scratch.nqp; ++q)
    {
      for (unsigned int i = 0; i < data.vel_dpc; ++i)
        for (unsigned int j = 0; j < data.pres_dpc; ++j)
          data.local_grad(i, j) += -scratch.fe_val_vel.JxW(q) *
                                    scratch.fe_val_pres.shape_value(j, q) *
                                    scratch.fe_val_vel[velocity].divergence(i, q);
    }
}

template <int dim>
void NavierStokesProjection<dim>::copy_gradient_local_to_global
(const InitGradPerTaskData &data)
{
  for (unsigned int i = 0; i < data.vel_dpc; ++i)
    for (unsigned int j = 0; j < data.pres_dpc; ++j)
      pres_Diff.add(data.vel_local_dof_indices[i],
                    data.pres_local_dof_indices[j],
                    data.local_grad(i, j));
}

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::initialize_gradient_operator();
template void Step35::NavierStokesProjection<3>::initialize_gradient_operator();

template void Step35::NavierStokesProjection<2>::assemble_one_cell_of_gradient
(const IteratorPair &,
 InitGradScratchData &,
 InitGradPerTaskData &);
template void Step35::NavierStokesProjection<3>::assemble_one_cell_of_gradient
(const IteratorPair &,
 InitGradScratchData &,
 InitGradPerTaskData &);

template void Step35::NavierStokesProjection<2>::copy_gradient_local_to_global
(const InitGradPerTaskData &);
template void Step35::NavierStokesProjection<3>::copy_gradient_local_to_global
(const InitGradPerTaskData &);
