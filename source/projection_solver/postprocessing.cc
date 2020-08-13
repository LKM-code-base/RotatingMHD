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
postprocessing()
{
  dfg_benchmark.compute_pressure_difference(pressure_dof_handler,
                                            pressure_n);
  pcout << dfg_benchmark.pressure_difference << std::endl;
}

}  // namespace Step35

// explicit instantiations
template void Step35::NavierStokesProjection<2>::postprocessing();
template void Step35::NavierStokesProjection<3>::postprocessing();
