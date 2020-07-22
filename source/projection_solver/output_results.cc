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
void NavierStokesProjection<dim>::output_results(const unsigned int step)
{
  std::vector<std::string> names(dim, "velocity");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(dim,
                           DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim>        data_out;
  data_out.add_data_vector(dof_handler_velocity, u_n, names, component_interpretation);
  data_out.add_data_vector(dof_handler_pressure, pres_n, "Pressure");
  data_out.build_patches(deg);

  std::ofstream output_file("solution-"
                            + Utilities::int_to_string(step, 5)
                            + ".vtk");
  data_out.write_vtk(output_file);
}

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::output_results(const unsigned int);
template void Step35::NavierStokesProjection<3>::output_results(const unsigned int);
