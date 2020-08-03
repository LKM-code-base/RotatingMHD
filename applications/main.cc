/*
 * main.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/projection_solver.h>

#include <iostream>

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace Step35;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(
        argc, argv, 1);

      RunTimeParameters::ParameterSet parameter_set;
      parameter_set.read_data_from_file("parameter_file.prm");

      deallog.depth_console(parameter_set.flag_verbose_output ? 2 : 0);

      NavierStokesProjection<2> simulation(parameter_set);
      simulation.run(parameter_set.flag_verbose_output, 
                     parameter_set.output_interval);
  }
  catch (std::exception &exc)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  catch (...)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!" << std::endl
            << "Don't forget to brush your teeth :-)" << std::endl
            << std::endl;
  return 0;
}
