#include <rotatingMHD/hydrodynamic_problem.h>

#include <deal.II/numerics/data_out.h>

namespace RMHD
{

template<int dim>
HydrodynamicProblem<dim>::HydrodynamicProblem
(const RunTimeParameters::HydrodynamicProblemParameters &prm)
:
Problem<dim>(prm),
parameters(prm),
velocity(std::make_shared<Entities::VectorEntity<dim>>
         (prm.fe_degree_velocity,
          this->triangulation,
          "Velocity")),
pressure(std::make_shared<Entities::ScalarEntity<dim>>
         (prm.fe_degree_pressure,
          this->triangulation,
          "Pressure")),
time_stepping(prm.time_discretization_parameters),
navier_stokes(prm.navier_stokes_parameters,
              time_stepping,
              velocity,
              pressure,
              this->mapping,
              this->pcout,
              this->computing_timer),
cfl_number(0.0)
{
  *this->pcout << parameters << std::endl << std::endl;

  this->container.add_entity(velocity);
  this->container.add_entity(pressure, false);
  this->container.add_entity(navier_stokes.phi, false);
}

template<int dim>
void HydrodynamicProblem<dim>::run()
{
  this->make_grid();

  setup_dofs();

  this->setup_boundary_conditions();

  this->setup_initial_conditions();

  // modify member variable of RMHD::TimeDiscretization::DiscreteTime if necessary
  if (time_stepping.get_end_time() != parameters.time_discretization_parameters.final_time)
      time_stepping.set_end_time(parameters.time_discretization_parameters.final_time);

  const unsigned int n_maximum_steps = parameters.time_discretization_parameters.n_maximum_steps;

  *this->pcout << "Solving the problem until t = "
               << Utilities::to_string(time_stepping.get_end_time())
               << " or until "
               << Utilities::int_to_string(n_maximum_steps)
               << " time steps are performed"
               << std::endl;

  time_loop(n_maximum_steps);
}

template<int dim>
void HydrodynamicProblem<dim>::continue_run()
{
  const double final_time = parameters.time_discretization_parameters.final_time;

  AssertThrow(time_stepping.get_current_time() < final_time,
              ExcMessage("Current time is larger equal than the final time. The run "
                         "cannot be continued."));
  AssertThrow(time_stepping.get_step_number() < parameters.time_discretization_parameters.n_maximum_steps,
              ExcMessage("The continuation of the run is aborted because the "
                         "maximum number was reached!"));
  // modify member variable end_time of RMHD::TimeDiscretization::DiscreteTime
  time_stepping.set_end_time(final_time);

  // compute the number of remaining steps
  const unsigned int n_maximum_steps = parameters.time_discretization_parameters.n_maximum_steps;
  AssertThrow(n_maximum_steps > time_stepping.get_step_number(),
              ExcLowerRange(n_maximum_steps, time_stepping.get_step_number()));

  const unsigned int n_remaining_steps = n_maximum_steps - time_stepping.get_step_number();

  *this->pcout << "Continuing the current run until t = "
               << Utilities::to_string(time_stepping.get_end_time())
               << " or until "
               << Utilities::int_to_string(n_remaining_steps)
               << " time steps are performed"
               << std::endl;

  time_loop(n_remaining_steps);

}

template <int dim>
void HydrodynamicProblem<dim>::time_loop(const unsigned int n_steps)
{
  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Update the time step, i.e., sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
        this->compute_next_time_step(time_stepping, cfl_number));

    *this->pcout << time_stepping << std::endl;

    // Update the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e., computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    time_stepping.advance_time();
    update_solution_vectors();

    // Snapshot stage
    if (parameters.postprocessing_frequency > 0)
      if (time_stepping.get_step_number() % parameters.postprocessing_frequency == 0 ||
          time_stepping.get_current_time() == time_stepping.get_end_time())
        this->postprocess_solution();

    if (parameters.spatial_discretization_parameters.adaptive_mesh_refinement)
      if (time_stepping.get_step_number() %
          parameters.spatial_discretization_parameters.adaptive_mesh_refinement_frequency == 0)
        this->adaptive_mesh_refinement();

    if (parameters.graphical_output_frequency > 0)
      if ((time_stepping.get_step_number() % parameters.graphical_output_frequency == 0) ||
          (time_stepping.get_current_time() == time_stepping.get_end_time()))
        output_results();

    if (time_stepping.get_step_number() >= n_steps)
      break;
  }

  *this->pcout << time_stepping << std::endl << std::endl;

  this->save_postprocessing_results();

  if (time_stepping.get_current_time() == time_stepping.get_end_time())
    *this->pcout << std::endl << std::endl
                 << "This run completed successfully!" << std::endl << std::endl;
  else if (time_stepping.get_step_number() >= n_steps)
    *this->pcout << std::endl << std::endl
                 << std::setw(80)
                 << "This run terminated because the maximum number of steps was "
                    "reached! The current\n time is not equal to the desired "
                    "final time." << std::endl << std::endl;

  *(this->pcout) << std::fixed;
}

template<int dim>
void HydrodynamicProblem<dim>::restart(const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}

template <int dim>
void HydrodynamicProblem<dim>::initialize_from_function
(Function<dim> &velocity_function,
 Function<dim> &pressure_function,
 const double   previous_step_size)
{
  Assert(previous_step_size > 0,
         ExcLowerRangeType<double>(previous_step_size, 0));

  Assert(this->time_stepping.get_step_number() == 0,
         ExcMessage("Initialization is not performed at the start of the simulation."));
  const double current_time = this->time_stepping.get_current_time();

  // compute two fictitious previous times
  const double previous_time = current_time - previous_step_size;

  // initialize previous solutions of the velocity
  {
    velocity_function.set_time(previous_time);
    this->project_function(velocity_function,
                           this->velocity,
                           this->velocity->old_old_solution);

    velocity_function.set_time(current_time);
    this->project_function(velocity_function,
                           this->velocity,
                           this->velocity->old_solution);
  }
  // initialize previous solutions of the pressure
  {
    pressure_function.set_time(previous_time);
    this->project_function(pressure_function,
                           this->pressure,
                           this->pressure->old_old_solution);

    pressure_function.set_time(current_time);
    this->project_function(pressure_function,
                           this->pressure,
                           this->pressure->old_solution);
  }

  // initialize the coefficients of the IMEX scheme
  //  time_stepping.initialize(previous_step_size);
}

template<int dim>
void HydrodynamicProblem<dim>::clear()
{
  navier_stokes.clear();
//  time_stepping.clear();

  velocity->clear();
  pressure->clear();

  Problem<dim>::clear();
}


template<int dim>
void HydrodynamicProblem<dim>::deserialize(const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}

template<int dim>
void HydrodynamicProblem<dim>::serialize(const std::string &) const
{
  AssertThrow(false, ExcNotImplemented());
}

template<int dim>
void HydrodynamicProblem<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();

  pressure->setup_dofs();

  *this->pcout << "Number of velocity degrees of freedom = "
               << (velocity->dof_handler)->n_dofs()
               << std::endl
               << "Number of pressure degrees of freedom = "
               << (pressure->dof_handler)->n_dofs()
               << std::endl
               << "Number of total degrees of freedom    = "
               << (pressure->dof_handler->n_dofs() +
                  velocity->dof_handler->n_dofs())
               << std::endl << std::endl;
}

template <int dim>
void HydrodynamicProblem<dim>::output_results() const
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(dim,
                           DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim>        data_out;

  data_out.add_data_vector(*(velocity->dof_handler),
                           velocity->solution,
                           names,
                           component_interpretation);

  data_out.add_data_vector(*(pressure->dof_handler),
                           pressure->solution,
                           "Pressure");

  data_out.build_patches(velocity->fe_degree);

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(this->prm.graphical_output_directory,
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);

  out_index++;
}

} // namespace RMHD

// explicit instantiations
template class RMHD::HydrodynamicProblem<2>::HydrodynamicProblem;
template class RMHD::HydrodynamicProblem<3>::HydrodynamicProblem;
