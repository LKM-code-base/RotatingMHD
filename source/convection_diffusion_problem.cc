#include <rotatingMHD/convection_diffusion_problem.h>

#include <deal.II/numerics/data_out.h>

#include <fstream>

namespace RMHD
{

template<int dim>
ConvectionDiffusionProblem<dim>::ConvectionDiffusionProblem
(ConvectionDiffusionProblemParameters &prm,
 const std::string  &field_name)
:
Problem<dim>(static_cast<RunTimeParameters::ProblemBaseParameters>(prm)),
parameters(prm),
scalar_field(std::make_shared<Entities::ScalarEntity<dim>>
            (prm.fe_degree,
             this->triangulation,
             field_name)),
time_stepping(prm.time_discretization_parameters),
solver(prm.parameters,
       time_stepping,
       scalar_field,
       this->mapping,
       this->pcout,
       this->computing_timer),
cfl_number(0.0)
{
  *this->pcout << parameters << std::endl << std::endl;

  this->container.add_entity(scalar_field);
}

template<int dim>
void ConvectionDiffusionProblem<dim>::run()
{
  this->make_grid();

  setup_dofs();

  this->setup_boundary_conditions();

  this->setup_initial_conditions();

  // modify member variable of RMHD::TimeDiscretization::DiscreteTime if necessary
  const double desired_end_time{parameters.time_discretization_parameters.final_time};
  if (time_stepping.get_end_time() != desired_end_time)
      time_stepping.set_end_time(desired_end_time);

  const unsigned int n_maximum_steps{parameters.time_discretization_parameters.n_maximum_steps};

  *this->pcout << "Solving the problem until t = "
               << Utilities::to_string(time_stepping.get_end_time())
               << " or until "
               << Utilities::int_to_string(n_maximum_steps)
               << " time steps are performed"
               << std::endl;

  time_loop(n_maximum_steps);
}



template<int dim>
void ConvectionDiffusionProblem<dim>::continue_run()
{
  const double desired_end_time{parameters.time_discretization_parameters.final_time};
  const unsigned int n_maximum_steps{parameters.time_discretization_parameters.n_maximum_steps};

  AssertThrow(time_stepping.get_current_time() < desired_end_time,
              ExcMessage("Current time is larger equal than the final time. The run "
                         "cannot be continued."));
  AssertThrow(time_stepping.get_step_number() < n_maximum_steps,
              ExcMessage("The continuation of the run is aborted because the "
                         "maximum number was reached!"));
  // modify member variable end_time of RMHD::TimeDiscretization::DiscreteTime
  time_stepping.set_end_time(desired_end_time);

  // compute the number of remaining steps

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
void ConvectionDiffusionProblem<dim>::time_loop(const unsigned int n_steps)
{
  // Copy parameters for better readability of the code
  const bool adaptive_mesh_refinement{parameters.spatial_discretization_parameters.adaptive_mesh_refinement};
  const unsigned int refinement_frequency{parameters.spatial_discretization_parameters.adaptive_mesh_refinement_frequency};
  const unsigned int postprocessing_frequency{parameters.postprocessing_frequency};
  const unsigned int terminal_output_frequency{parameters.terminal_output_frequency};
  const unsigned int graphical_output_frequency{parameters.graphical_output_frequency};

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    //  cfl_number = solver.get_cfl_number();

    // Update the time step, i.e., sets the value of t^{k}
    const double desired_next_step_size{this->compute_next_time_step(time_stepping, cfl_number)};
    time_stepping.set_desired_next_step_size(desired_next_step_size);

    if (this->prm.verbose)
      *this->pcout << time_stepping << std::endl;

    // Update the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e., computes the fields at t^{k}
    solver.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage
    if (postprocessing_frequency > 0)
      if (time_stepping.get_step_number() % postprocessing_frequency == 0 ||
          time_stepping.get_current_time() == time_stepping.get_end_time())
        this->postprocess_solution();

    if (adaptive_mesh_refinement)
      if (time_stepping.get_step_number() % refinement_frequency == 0)
        this->adaptive_mesh_refinement();

    if (graphical_output_frequency > 0)
      if ((time_stepping.get_step_number() % graphical_output_frequency == 0) ||
          (time_stepping.get_current_time() == time_stepping.get_end_time()))
        this->output_results();

    if (time_stepping.get_step_number() >= n_steps)
      break;
  }

  *this->pcout << time_stepping << std::endl;

  this->save_postprocessing_results();

  if (time_stepping.get_current_time() == time_stepping.get_end_time())
    *this->pcout << "This run completed successfully!" << std::endl << std::endl;
  else if (time_stepping.get_step_number() >= n_steps)
    *this->pcout << std::setw(80)
                 << "This run terminated because the maximum number of steps was "
                    "reached! The current\n time is not equal to the desired "
                    "final time." << std::endl << std::endl;

  *(this->pcout) << std::fixed;
}

template<int dim>
void ConvectionDiffusionProblem<dim>::restart(const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}

template <int dim>
void ConvectionDiffusionProblem<dim>::initialize_from_function
(Function<dim> &function,
 const double   previous_step_size)
{
  Assert(previous_step_size > 0,
         ExcLowerRangeType<double>(previous_step_size, 0));

  Assert(this->time_stepping.get_step_number() == 0,
         ExcMessage("Initialization is not performed at the start of the simulation."));
  const double current_time{this->time_stepping.get_current_time()};

  // compute two fictitious previous times
  const double previous_time{current_time - previous_step_size};

  // initialize previous solutions of the velocity
  {
    function.set_time(previous_time);
    this->project_function(function,
                           this->scalar_field,
                           this->scalar_field->old_old_solution);

    function.set_time(current_time);
    this->project_function(function,
                           this->scalar_field,
                           this->scalar_field->old_solution);
  }

  // initialize the coefficients of the IMEX scheme
  time_stepping.initialize(previous_step_size);
}

template<int dim>
void ConvectionDiffusionProblem<dim>::clear()
{
  solver.clear();

  time_stepping.clear();

  scalar_field->clear();

  Problem<dim>::clear();
}


template<int dim>
void ConvectionDiffusionProblem<dim>::deserialize(const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}

template<int dim>
void ConvectionDiffusionProblem<dim>::serialize(const std::string &) const
{
  AssertThrow(false, ExcNotImplemented());
}

template<int dim>
void ConvectionDiffusionProblem<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  scalar_field->setup_dofs();

  *this->pcout << "Number of " << scalar_field->name << " degrees of freedom = "
               << (scalar_field->dof_handler)->n_dofs()
               << std::endl << std::endl;
}

template <int dim>
void ConvectionDiffusionProblem<dim>::output_results() const
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  DataOut<dim>        data_out;
  data_out.add_data_vector(*(scalar_field->dof_handler),
                           scalar_field->solution,
                           scalar_field->name);
  data_out.build_patches(scalar_field->fe_degree);

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
template class RMHD::ConvectionDiffusionProblem<2>;
template class RMHD::ConvectionDiffusionProblem<3>;
