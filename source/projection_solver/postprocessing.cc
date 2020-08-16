/*
 * setup.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <iostream>
#include <fstream>

namespace Step35 
{
template <int dim>
void NavierStokesProjection<dim>::
postprocessing(const unsigned int &step,
               const unsigned int &terminal_output_interval)
{
  if (flag_DFG_benchmark)
  {
    dfg_benchmark.compute_pressure_difference(pressure_dof_handler,
                                              pressure_n);
    dfg_benchmark.compute_drag_and_lift_forces_and_coefficients(
                                              triangulation,
                                              velocity_fe,
                                              velocity_fe_degree,
                                              velocity_dof_handler,
                                              velocity_n,
                                              pressure_fe,
                                              pressure_dof_handler,
                                              pressure_n,
                                              Re);
    dfg_benchmark.update_table(step,
                               time_stepping);
    if ((step % terminal_output_interval == 0) || 
         time_stepping.is_at_end())
      dfg_benchmark.print_step_data(step,
                                    time_stepping);
    dfg_benchmark.write_table_to_file("table.tex");
  }
}

}  // namespace Step35

// explicit instantiations
template void Step35::NavierStokesProjection<2>::postprocessing(
                                                  const unsigned int &,
                                                  const unsigned int &);
template void Step35::NavierStokesProjection<3>::postprocessing(
                                                  const unsigned int &,
                                                  const unsigned int &);
