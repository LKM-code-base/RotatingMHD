/*
 * run_time_parameters.cc
 *
 *  Created on: Nov 22, 2020
 *      Author: sg
 */

#include <rotatingMHD/run_time_parameters.h>

int main(int /* argc */, char **/* argv[] */)
{
  try
  {
      using namespace RMHD::RunTimeParameters;

      ProblemParameters problem_params;
      problem_params.adaptive_mesh_refinement_frequency = 123456789;
      problem_params.graphical_output_directory = "./foobar/";
      problem_params.initial_time_step = 1000.123456;

      NavierStokesDiscretizationParameters ns_discretization_params;
      ns_discretization_params.preconditioner_update_frequency = 123;
      ns_discretization_params.projection_method = ProjectionMethod::standard;

      const NavierStokesProblemParameters ns_problem_params;

      std::cout << "======= Problem parameters ================================"
                << std::endl;
      std::cout << problem_params << std::endl;
      std::cout << "==========================================================="
                << std::endl;

      std::cout << "======= Navier-Stokes discretization parameters ==========="
                << std::endl;
      std::cout << ns_discretization_params << std::endl;
      std::cout << "==========================================================="
                << std::endl;

      std::cout << "======= Navier-Stokes problem parameters =================="
                << std::endl;
      std::cout << ns_problem_params << std::endl;
      std::cout << "==========================================================="
                << std::endl;

  }
  catch (std::exception &exc)
  {
      std::cerr << std::endl << std::endl
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
      std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
      std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
      return 1;
  }
  return 0;
}


