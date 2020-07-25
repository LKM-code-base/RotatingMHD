#include <rotatingMHD/projection_solver.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/fe/component_mask.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>

namespace Step35
{
template <int dim>
void NavierStokesProjection<dim>::
setup_dofs()
{
  velocity_dof_handler.distribute_dofs(velocity_fe);
  DoFRenumbering::boost::Cuthill_McKee(velocity_dof_handler);
  pressure_dof_handler.distribute_dofs(pressure_fe);
  DoFRenumbering::boost::Cuthill_McKee(pressure_dof_handler);

  {
    velocity_constraints.clear();
    DoFTools::make_hanging_node_constraints(velocity_dof_handler,
                                            velocity_constraints);
    for (const auto &boundary_id : boundary_ids)
    {
      switch (boundary_id)
      {
        case 1:
          VectorTools::interpolate_boundary_values(
                                      velocity_dof_handler,
                                      boundary_id,
                                      Functions::ZeroFunction<dim>(dim),
                                      velocity_constraints);
          break;
        case 2:
          VectorTools::interpolate_boundary_values(
                                      velocity_dof_handler,
                                      boundary_id,
                                      inflow_boundary_condition,
                                      velocity_constraints);
          break;
        case 3:
        {
          ComponentMask   component_mask(dim, true);
          component_mask.set(0, false);
          VectorTools::interpolate_boundary_values(
                                      velocity_dof_handler,
                                      boundary_id,
                                      Functions::ZeroFunction<dim>(dim),
                                      velocity_constraints,
                                      component_mask);
          break;
        }
        case 4:
          VectorTools::interpolate_boundary_values(
                                      velocity_dof_handler,
                                      boundary_id,
                                      Functions::ZeroFunction<dim>(dim),
                                      velocity_constraints);
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
    }
    velocity_constraints.close();
  }

  {
    pressure_constraints.clear();
    DoFTools::make_hanging_node_constraints(pressure_dof_handler,
                                            pressure_constraints);
    VectorTools::interpolate_boundary_values(
                                        pressure_dof_handler,
                                        3,
                                        Functions::ZeroFunction<dim>(),
                                        pressure_constraints);
    pressure_constraints.close();
  }

  std::cout << "Number of velocity degrees of freedom = " 
            << velocity_dof_handler.n_dofs()
            << std::endl
            << "Number of pressure degrees of freedom = " 
            << pressure_dof_handler.n_dofs()
            << std::endl
            << "Reynolds number                       = " 
            << Re << std::endl << std::endl;
}
}

// explicit instantiations
template void Step35::NavierStokesProjection<2>::setup_dofs();
template void Step35::NavierStokesProjection<3>::setup_dofs();
