/*
 * timestepping_coefficients.cc
 *
 *  Created on: Jul 23, 2019
 *      Author: sg
 */
#include <iostream>
#include <exception>
#include <functional>
#include <vector>

#include <rotatingMHD/time_discretization.h>

using namespace RMHD::TimeDiscretization;

void checkTimeStepper(const TimeSteppingParameters &parameters)
{
    VSIMEXMethod  timestepper(parameters);

    if (parameters.adaptive_time_stepping)
    {
        std::cout << "Adaptive time stepping with "
                  << timestepper.get_name() << " scheme" << std::endl;

        const std::vector<double> timesteps({0.1,0.1,0.1,0.05,0.15,0.9});

        do
        {
          std::cout << timestepper << std::endl;

          timestepper.update_coefficients();

          timestepper.print_coefficients(std::cout);

          timestepper.set_desired_next_step_size(timesteps[timestepper.get_step_number()]);

          timestepper.advance_time();

        } while (timestepper.is_at_end() == false &&
                 timestepper.get_step_number() < parameters.n_maximum_steps);

    }
    else
    {
        std::cout << "Fixed time stepping with "
                  << timestepper.get_name() << " scheme" << std::endl;

        do
        {
          std::cout << timestepper << std::endl;

          timestepper.update_coefficients();

          timestepper.print_coefficients(std::cout);

          timestepper.advance_time();

        } while (timestepper.is_at_end() == false &&
                 timestepper.get_step_number() < parameters.n_maximum_steps);
    }

    return;
}

int main(int /* argc */, char **/* argv[] */)
{
  try
  {
    TimeSteppingParameters parameters("timestepping_coefficients.prm");

    parameters.final_time = 3.0;
    parameters.initial_time_step = 1.0;
    parameters.maximum_time_step = 1.0;
    parameters.minimum_time_step = 0.001;

    // fixed time steps
    parameters.adaptive_time_stepping = false;

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::ForwardEuler;
    checkTimeStepper(parameters);

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::BDF2;
    checkTimeStepper(parameters);

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::CNAB;
    checkTimeStepper(parameters);

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::mCNAB;
    checkTimeStepper(parameters);

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::CNLF;
    checkTimeStepper(parameters);


    // adaptive time steps
    parameters.final_time = 1.0;
    parameters.initial_time_step = 0.1;
    parameters.maximum_time_step = 1.0;
    parameters.minimum_time_step = 0.001;

    /*
     * To be performed when the adaptivity is implemented...
     *
    parameters.adaptive_time_stepping = true;

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::BDF2;
    checkTimeStepper(parameters);

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::CNAB;
    checkTimeStepper(parameters);

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::mCNAB;
    checkTimeStepper(parameters);

    std::cout << "================================="
                 "========================================" << std::endl;
    parameters.vsimex_scheme = VSIMEXScheme::CNLF;
    checkTimeStepper(parameters);

    *
    */
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
