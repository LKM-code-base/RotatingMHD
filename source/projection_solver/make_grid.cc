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

namespace Step35 
{
  template <int dim>
  void NavierStokesProjection<dim>::
  make_grid(const unsigned int n_global_refinements)
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);

    {
      std::string   filename = "nsbench2.inp";
      std::ifstream file(filename);
      Assert(file, ExcFileNotOpen(filename.c_str()));
      grid_in.read_ucd(file);
    }
    triangulation.refine_global(n_global_refinements);
    boundary_ids = triangulation.get_boundary_ids();

    std::cout << "Number of refines = " 
              << n_global_refinements << std::endl;
    std::cout << "Number of active cells: " 
              << triangulation.n_active_cells() << std::endl;
  }

}  // namespace Step35

// explicit instantiations

template void Step35::NavierStokesProjection<2>::make_grid(const unsigned int);
template void Step35::NavierStokesProjection<3>::make_grid(const unsigned int);
