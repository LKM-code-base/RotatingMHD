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
    if (time_stepping.get_step_number() %
         this->prm.terminal_output_frequency == 0 ||
        time_stepping.get_current_time() == time_stepping.get_end_time())
      this->postprocess_solution();

    if (time_stepping.get_step_number() %
        this->prm.spatial_discretization_parameters.adaptive_mesh_refinement_frequency == 0)
      this->adaptive_mesh_refinement();

    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
                   time_stepping.get_end_time()))
      output_results();
  }

  *(this->pcout) << std::fixed;

  clear();
}

template<int dim>
void HydrodynamicProblem<dim>::continue_run()
{
  AssertThrow(false, ExcNotImplemented());
}

template<int dim>
void HydrodynamicProblem<dim>::restart(const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}

template<int dim>
void HydrodynamicProblem<dim>::restart_from_function
(const Function<dim> &,
 const double,
 const double)
{
  AssertThrow(false, ExcNotImplemented());
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
