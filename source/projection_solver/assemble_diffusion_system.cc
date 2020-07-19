/*
 * assemble_diffusion_system.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/base/work_stream.h>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::assemble_advection_term()
{
  vel_Advection = 0.;

  AdvectionPerTaskData data(fe_velocity.dofs_per_cell);

  AdvectionScratchData scratch(fe_velocity,
                               quadrature_velocity,
                               update_values | update_JxW_values |
                               update_gradients);

  WorkStream::run(dof_handler_velocity.begin_active(),
                  dof_handler_velocity.end(),
                  *this,
                  &NavierStokesProjection<dim>::assemble_one_cell_of_advection,
                  &NavierStokesProjection<dim>::copy_advection_local_to_global,
                  scratch,
                  data);
}

template <int dim>
void NavierStokesProjection<dim>::assemble_one_cell_of_advection
(const typename DoFHandler<dim>::active_cell_iterator &cell,
 AdvectionScratchData &                                scratch,
 AdvectionPerTaskData &                                data)
{
  scratch.fe_val.reinit(cell);
  cell->get_dof_indices(data.local_dof_indices);
  const FEValuesExtractors::Vector      velocity(0);
  //for (unsigned int d = 0; d < dim; ++d)
  //  {
  scratch.fe_val[velocity].get_function_values(u_star, scratch.u_star_local);
  //for (unsigned int q = 0; q < scratch.nqp; ++q)
  //  scratch.u_star_local[q](d) = scratch.u_star_tmp[q];
  //  }

  //for (unsigned int d = 0; d < dim; ++d)
  //  {
  scratch.fe_val[velocity].get_function_divergences(u_star, scratch.div_u_star);
  //for (unsigned int q = 0; q < scratch.nqp; ++q)
  //  {
  //    if (d == 0)
  //      scratch.u_star_tmp[q] = 0.;
  //    scratch.u_star_tmp[q] += scratch.grad_u_star[q][d];
  //  }
  //  }

  data.local_advection = 0.;
  for (unsigned int q = 0; q < scratch.nqp; ++q)
    for (unsigned int i = 0; i < scratch.dpc; ++i)
      for (unsigned int j = 0; j < scratch.dpc; ++j)
        data.local_advection(i, j) += ( scratch.fe_val[velocity].value(i, q) *
                                        scratch.fe_val[velocity].gradient(j, q) *
                                        scratch.u_star_local[q]
                                      + 0.5 * scratch.div_u_star[q] *
                                        scratch.fe_val[velocity].value(i, q) *
                                        scratch.fe_val[velocity].value(j, q)
                                      )
                                      * scratch.fe_val.JxW(q);
}

template <int dim>
void NavierStokesProjection<dim>::copy_advection_local_to_global
(const AdvectionPerTaskData &data)
{
  for (unsigned int i = 0; i < fe_velocity.dofs_per_cell; ++i)
    for (unsigned int j = 0; j < fe_velocity.dofs_per_cell; ++j)
      vel_Advection.add(data.local_dof_indices[i],
                        data.local_dof_indices[j],
                        data.local_advection(i, j));
}

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::assemble_advection_term();
template void Step35::NavierStokesProjection<3>::assemble_advection_term();

template void Step35::NavierStokesProjection<2>::assemble_one_cell_of_advection
(const typename DoFHandler<2>::active_cell_iterator &,
 AdvectionScratchData &,
 AdvectionPerTaskData &);
template void Step35::NavierStokesProjection<3>::assemble_one_cell_of_advection
(const typename DoFHandler<3>::active_cell_iterator &,
 AdvectionScratchData &,
 AdvectionPerTaskData &);

template void Step35::NavierStokesProjection<2>::copy_advection_local_to_global(const AdvectionPerTaskData &);
template void Step35::NavierStokesProjection<3>::copy_advection_local_to_global(const AdvectionPerTaskData &);
