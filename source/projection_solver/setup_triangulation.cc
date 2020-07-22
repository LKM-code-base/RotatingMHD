/*
 * setup.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/grid/grid_in.h>

#include <iostream>
#include <fstream>

namespace Step35 {

template <int dim>
void NavierStokesProjection<dim>::create_triangulation_and_dofs(const unsigned int n_refines)
{
  std::cout << "Reading grid..." << std::endl;

  GridIn<dim> grid_in;
  grid_in.attach_triangulation(triangulation);

  {
    std::string   filename = "nsbench2.inp";
    std::ifstream file(filename);
    Assert(file, ExcFileNotOpen(filename.c_str()));
    grid_in.read_ucd(file);
  }

  std::cout << "   Number of refines = " << n_refines << std::endl;
  triangulation.refine_global(n_refines);
  std::cout << "   Number of active cells: " << triangulation.n_active_cells()
            << std::endl;

  boundary_ids = triangulation.get_boundary_ids();

  std::cout << "Setup dofs..." << std::endl;

  dof_handler_velocity.distribute_dofs(fe_velocity);
  DoFRenumbering::boost::Cuthill_McKee(dof_handler_velocity);

  dof_handler_pressure.distribute_dofs(fe_pressure);
  DoFRenumbering::boost::Cuthill_McKee(dof_handler_pressure);

  std::cout << "   dim (X_h) = " << (dof_handler_velocity.n_dofs() )
            << std::endl
            << "   dim (M_h) = " << dof_handler_pressure.n_dofs()
            << std::endl
            << "   Re        = " << Re << std::endl
            << std::endl;

  std::cout << "Setup sparsity patterns..." << std::endl;

  initialize_velocity_matrices();
  initialize_pressure_matrices();
  initialize_gradient_operator();

  pres_n.reinit(dof_handler_pressure.n_dofs());
  pres_n_minus_1.reinit(dof_handler_pressure.n_dofs());
  phi_n.reinit(dof_handler_pressure.n_dofs());
  phi_n_minus_1.reinit(dof_handler_pressure.n_dofs());
  pres_tmp.reinit(dof_handler_pressure.n_dofs());

  //for (unsigned int d = 0; d < dim; ++d)
  //  {
  u_n.reinit(dof_handler_velocity.n_dofs());
  u_n_minus_1.reinit(dof_handler_velocity.n_dofs());
  u_star.reinit(dof_handler_velocity.n_dofs());
  force.reinit(dof_handler_velocity.n_dofs());
  //  }

  v_tmp.reinit(dof_handler_velocity.n_dofs());
  rot_u.reinit(dof_handler_velocity.n_dofs());
}

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::create_triangulation_and_dofs(const unsigned int);
template void Step35::NavierStokesProjection<3>::create_triangulation_and_dofs(const unsigned int);
