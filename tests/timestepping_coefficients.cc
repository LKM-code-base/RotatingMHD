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

void checkTimeStepper(const TimeDiscretizationParameters &parameters)
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

        while (timestepper.get_current_time() < timestepper.get_end_time())
        {
          // Update the time step, i.e., sets the value of t^{k}
          timestepper.set_desired_next_step_size(1.0);

          std::cout << timestepper << std::endl;

          // Update the coefficients to their k-th value
          timestepper.update_coefficients();

          // Print the coefficients
          timestepper.print_coefficients(std::cout);

          // Advances the VSIMEXMethod instance to t^{k}
          timestepper.advance_time();

          if (timestepper.get_step_number() >= parameters.n_maximum_steps)
            break;
        }

        std::cout << "Restarting time stepping with "
                  << timestepper.get_name() << " scheme" << std::endl;

        timestepper.clear();
        timestepper.initialize(1.0);

        while (timestepper.get_current_time() < timestepper.get_end_time())
        {
          // Update the time step, i.e., sets the value of t^{k}
          timestepper.set_desired_next_step_size(1.0);

          std::cout << timestepper << std::endl;

          // Update the coefficients to their k-th value
          timestepper.update_coefficients();

          // Print the coefficients
          timestepper.print_coefficients(std::cout);

          // Advances the VSIMEXMethod instance to t^{k}
          timestepper.advance_time();

          if (timestepper.get_step_number() >= parameters.n_maximum_steps)
            break;
        }

    }
    return;
}

int main(int /* argc */, char **/* argv[] */)
{
  try
  {
    TimeDiscretizationParameters parameters;

    parameters.final_time = 4.0;
    parameters.initial_time_step = 1.0;
    parameters.maximum_time_step = 1.0;
    parameters.minimum_time_step = 0.1;

    // fixed time steps
    parameters.adaptive_time_stepping = false;

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
//    parameters.final_time = 1.0;
//    parameters.initial_time_step = 0.1;
//    parameters.maximum_time_step = 1.0;
//    parameters.minimum_time_step = 0.001;

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
