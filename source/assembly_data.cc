/*
 * assembly_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/assembly_data.h>

namespace Step35
{
  using namespace dealii;
  namespace AdvectionTermAssembly
  {
    template <int dim>
    MappingData<dim>::MappingData(const unsigned int dofs_per_cell)
        : local_matrix(dofs_per_cell, dofs_per_cell)
        , local_dof_indices(dofs_per_cell)
      {}

    template <int dim>  
    LocalCellData<dim>::LocalCellData(const FESystem<dim> &fe,
                    const QGauss<dim>   &quadrature_formula,
                    const UpdateFlags   flags)
        : fe_values(fe, quadrature_formula, flags),
          n_q_points(quadrature_formula.size()),
          dofs_per_cell(fe.dofs_per_cell),
          v_extrapolated_divergence(n_q_points),
          v_extrapolated_values(n_q_points),
          phi_v(dofs_per_cell),
          grad_phi_v(dofs_per_cell)
      {}

    template <int dim>
    LocalCellData<dim>::LocalCellData(const LocalCellData &data)
        : fe_values(data.fe_values.get_fe(),
                    data.fe_values.get_quadrature(),
                    data.fe_values.get_update_flags()),
          n_q_points(data.n_q_points),
          dofs_per_cell(data.dofs_per_cell),
          v_extrapolated_divergence(n_q_points),
          v_extrapolated_values(n_q_points),
          phi_v(dofs_per_cell),
          grad_phi_v(dofs_per_cell)
      {}

  }// namespace AdvectionTermAssembly

  namespace PressureGradientTermAssembly
  {
      template <int dim>
      MappingData<dim>::MappingData(const unsigned int v_dofs_per_cell,
                              const unsigned int p_dofs_per_cell)
          : v_dofs_per_cell(v_dofs_per_cell),
            p_dofs_per_cell(p_dofs_per_cell),
            local_matrix(v_dofs_per_cell, p_dofs_per_cell),
            v_local_dof_indices(v_dofs_per_cell),
            p_local_dof_indices(p_dofs_per_cell)
        {}

      template <int dim>  
      LocalCellData<dim>::LocalCellData(const FESystem<dim> &v_fe,
                      const FE_Q<dim>     &p_fe,
                      const QGauss<dim>   &quadrature_formula,
                      const UpdateFlags   v_flags,
                      const UpdateFlags   p_flags)
          : v_fe_values(v_fe, quadrature_formula, v_flags),
            p_fe_values(p_fe, quadrature_formula, p_flags),
            n_q_points(quadrature_formula.size()),
            v_dofs_per_cell(v_fe.dofs_per_cell),
            p_dofs_per_cell(p_fe.dofs_per_cell),
            div_phi_v(v_dofs_per_cell),
            phi_p(p_dofs_per_cell)
        {}

      template <int dim>
      LocalCellData<dim>::LocalCellData(const LocalCellData &data)
          : v_fe_values(data.v_fe_values.get_fe(),
                      data.v_fe_values.get_quadrature(),
                      data.v_fe_values.get_update_flags()),
            p_fe_values(data.p_fe_values.get_fe(),
                        data.p_fe_values.get_quadrature(),
                        data.p_fe_values.get_update_flags()),
            n_q_points(data.n_q_points),
            v_dofs_per_cell(data.v_dofs_per_cell),
            p_dofs_per_cell(data.v_dofs_per_cell),
            div_phi_v(v_dofs_per_cell),
            phi_p(p_dofs_per_cell)
        {}
  }// namespace PressureGradientTermAssembly

}

template struct Step35::AdvectionTermAssembly::MappingData<2>;
template struct Step35::AdvectionTermAssembly::MappingData<3>;
template struct Step35::AdvectionTermAssembly::LocalCellData<2>;
template struct Step35::AdvectionTermAssembly::LocalCellData<3>;
template struct Step35::PressureGradientTermAssembly::MappingData<2>;
template struct Step35::PressureGradientTermAssembly::MappingData<3>;
template struct Step35::PressureGradientTermAssembly::LocalCellData<2>;
template struct Step35::PressureGradientTermAssembly::LocalCellData<3>;