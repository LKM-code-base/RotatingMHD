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
    v_dof_handler.distribute_dofs(v_fe);
    DoFRenumbering::boost::Cuthill_McKee(v_dof_handler);
    p_dof_handler.distribute_dofs(p_fe);
    DoFRenumbering::boost::Cuthill_McKee(p_dof_handler);

    {
      v_constraints.clear();
      DoFTools::make_hanging_node_constraints(v_dof_handler,
                                              v_constraints);
      for (const auto &boundary_id : boundary_ids)
      {
        switch (boundary_id)
        {
          case 1:
            VectorTools::interpolate_boundary_values(
                                        v_dof_handler,
                                        boundary_id,
                                        Functions::ZeroFunction<dim>(dim),
                                        v_constraints);
            break;
          case 2:
            VectorTools::interpolate_boundary_values(
                                        v_dof_handler,
                                        boundary_id,
                                        inflow_bc,
                                        v_constraints);
            break;
          case 3:
          {
            ComponentMask   component_mask(dim, true);
            component_mask.set(0, false);
            VectorTools::interpolate_boundary_values(
                                        v_dof_handler,
                                        boundary_id,
                                        Functions::ZeroFunction<dim>(dim),
                                        v_constraints,
                                        component_mask);
            break;
          }
          case 4:
            VectorTools::interpolate_boundary_values(
                                        v_dof_handler,
                                        boundary_id,
                                        Functions::ZeroFunction<dim>(dim),
                                        v_constraints);
            break;
          default:
            Assert(false, ExcNotImplemented());
        }
      }
      v_constraints.close();
    }

    {
      p_constraints.clear();
      DoFTools::make_hanging_node_constraints(p_dof_handler,
                                              p_constraints);
      VectorTools::interpolate_boundary_values(p_dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(),
                                               p_constraints);
      p_constraints.close();
    }

    std::cout << "dim (X_h) = " << v_dof_handler.n_dofs()
              << std::endl
              << "dim (M_h) = " << p_dof_handler.n_dofs()
              << std::endl
              << "Re        = " << Re << std::endl
              << std::endl;
  }
}

template void Step35::NavierStokesProjection<2>::setup_dofs();
template void Step35::NavierStokesProjection<3>::setup_dofs();
