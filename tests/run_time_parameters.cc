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

      const ProblemParameters problem_params;

      const NavierStokesDiscretizationParameters ns_discretization_params;

      const NavierStokesProblemParameters ns_problem_params;

      std::cout << "======= Problem parameters ================================"
                << std::endl;
      std::cout << problem_params;
      std::cout << "==========================================================="
                << std::endl;

      std::cout << "======= Navier-Stokes discretization parameters ==========="
                << std::endl;
      std::cout << ns_discretization_params;
      std::cout << "==========================================================="
                << std::endl;

      std::cout << "======= Navier-Stokes problem parameters =================="
                << std::endl;
      std::cout << ns_problem_params;
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


