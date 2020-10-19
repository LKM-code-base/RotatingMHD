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
#include <cmath>

#include <rotatingMHD/time_discretization.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/lu.hpp>

using namespace RMHD::TimeDiscretization;

using namespace boost::numeric;

void checkTimeStepper(const TimeSteppingParameters &parameters,
                      const unsigned int n_points = 64,
                      const double viscosity = 0.1)
{
    assert(n_points > 0);
    /*
     * setting up an equidistantly spaced grid from x = 0 to x = 1
     */
    const double h = 1.0 / double(n_points - 1);

    ublas::vector<double>   points(n_points);
    for (unsigned int i=0; i<n_points; ++i)
        points(i) = i * h;
    assert(std::abs(points(0)) < 1e-12 );
    assert(std::abs(points(n_points-1) - 1.0) < 1e-12 );

    /*
     * setting up a variable (transport) velocity
     */
    ublas::vector<double>   velocity(n_points);
    for (unsigned int i=0; i<n_points; ++i)
        velocity[i] = std::sin(2.0*M_PI*points[i]);

    /*
     * finite difference matrices using periodic boundary conditions and
     * a second-order central difference scheme
     */
    const ublas::identity_matrix<double>    identity_matrix(n_points);

    ublas::matrix<double>   first_derivative_matrix(n_points, n_points);
    ublas::matrix<double>   second_derivative_matrix(n_points, n_points);

    // first rows are different due to periodic boundary conditions
    first_derivative_matrix(0,n_points-1) = -1.0 / (2.0 * h);
    first_derivative_matrix(0,1) = 1.0 / (2.0 * h);

    second_derivative_matrix(0,n_points-1) = 1.0 / (h * h);
    second_derivative_matrix(0,0) = -2.0 / (h * h);
    second_derivative_matrix(0,1) = 1.0 / (h * h);

    // other rows
    for (unsigned int i=1; i<(n_points-1); ++i)
    {
        first_derivative_matrix(i,i-1) = -1.0 / (2.0 * h);
        first_derivative_matrix(i,i+1) = 1.0 / (2.0 * h);

        second_derivative_matrix(i,i-1) = 1.0 / (h * h);
        second_derivative_matrix(i,i) = -2.0 / (h * h);
        second_derivative_matrix(i,i+1) = 1.0 / (h * h);
    }

    // last rows are different due to periodic boundary conditions
    first_derivative_matrix(n_points-1,n_points-2) = -1.0 / (2.0 * h);
    first_derivative_matrix(n_points-1,0) = 1.0 / (2.0 * h);

    second_derivative_matrix(n_points-1,n_points-2) = 1.0 / (h * h);
    second_derivative_matrix(n_points-1,n_points-1) = -2.0 / (h * h);
    second_derivative_matrix(n_points-1,0) = 1.0 / (h * h);

    /*
     * setting up the system matrix, solution and right-hand side
     */
    ublas::matrix<double>    system_matrix(n_points, n_points);
    ublas::permutation_matrix<std::size_t> permutation_matrix(n_points);
    ublas::vector<double>    solution(n_points), old_solution(n_points),
                             old_old_solution(n_points), system_rhs(n_points);

    /*
     * setting up the VSIMEXMethod object
     */
    VSIMEXMethod  timestepper(parameters);

    if (parameters.adaptive_time_stepping)
    {
        /*
         *
        std::cout << "Adaptive time stepping with "
                  << timestepper.get_name() << " scheme" << std::endl;

        do
        {
          std::cout << timestepper << std::endl;

          timestepper.update_coefficients();

          timestepper.print_coefficients(std::cout);

          timestepper.set_desired_next_step_size();

          timestepper.advance_time();

        } while (timestepper.is_at_end() == false &&
                 timestepper.get_step_number() < parameters.n_maximum_steps);
         *
         */
    }
    else
    {
        std::cout << "Fixed time stepping with "
                  << timestepper.get_name() << " scheme" << std::endl;

        system_matrix =
                  timestepper.get_alpha()[0] / timestepper.get_next_step_size() * identity_matrix
                + timestepper.get_gamma()[0] * second_derivative_matrix;

        ublas::lu_factorize(system_matrix, permutation_matrix);

        do
        {
          std::cout << timestepper << std::endl;

          timestepper.update_coefficients();

          const std::vector<double> alpha = timestepper.get_alpha();
          const std::vector<double> beta = timestepper.get_beta();
          const std::vector<double> gamma = timestepper.get_gamma();
          const double timestep_size = timestepper.get_next_step_size();
          /*
           * update the right-hand side
           */
          system_rhs =
                  (- alpha[1] * old_solution - alpha[2] * old_old_solution) / timestep_size
                  - beta[0] * ublas::element_prod(velocity, ublas::prod(first_derivative_matrix, old_solution))
                  - beta[1] * ublas::element_prod(velocity, ublas::prod(first_derivative_matrix, old_old_solution))
                  + viscosity * ( ublas::prod(second_derivative_matrix, old_solution)
                                + ublas::prod(second_derivative_matrix, old_old_solution));

          /*
           * solve the linear system
           */
          ublas::lu_substitute(system_matrix, permutation_matrix, system_rhs);
          solution = system_rhs;

          timestepper.advance_time();

          old_old_solution = old_solution;
          old_solution = solution;

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
