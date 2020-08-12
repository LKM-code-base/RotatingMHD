/*
 * output_results.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

namespace Step35
{

template <int dim>
void NavierStokesProjection<dim>::
output_results()
{
  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<dim>        data_out;

  data_out.add_data_vector(velocity_dof_handler, 
                           velocity_n, 
                           names, 
                           component_interpretation);
  data_out.add_data_vector(pressure_dof_handler, 
                           pressure_n, 
                           "Pressure");
  data_out.build_patches(pressure_fe_degree+1);
  
  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(
    "./", "solution", out_index, MPI_COMM_WORLD, 5);
  out_index++;
}

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::output_results();
template void Step35::NavierStokesProjection<3>::output_results();
