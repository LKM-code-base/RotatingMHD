#ifndef INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_
#define INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/fe/fe_q.h>

namespace Step35
{
  using namespace dealii;
  namespace AdvectionTermAssembly
  {
    template <int dim>
    struct MappingData
    {
      FullMatrix<double>                   local_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
      MappingData(const unsigned int dofs_per_cell);
    };

    template <int dim>  
    struct LocalCellData
    {
      FEValues<dim>               fe_values;
      unsigned int                n_q_points;
      unsigned int                dofs_per_cell;
      std::vector<double>         v_extrapolated_divergence;
      std::vector<Tensor<1,dim>>  v_extrapolated_values;
      std::vector<Tensor<1,dim>>  phi_v;
      std::vector<Tensor<2,dim>>  grad_phi_v;

      LocalCellData(const FESystem<dim> &fe,
                    const QGauss<dim>   &quadrature_formula,
                    const UpdateFlags   flags);

      LocalCellData(const LocalCellData &data);
    };
  }// namespace AdvectionTermAssembly

  namespace PressureGradientTermAssembly
  {
    template <int dim>
    struct MappingData
    {
      unsigned int                         v_dofs_per_cell;
      unsigned int                         p_dofs_per_cell;
      FullMatrix<double>                   local_matrix;
      std::vector<types::global_dof_index> v_local_dof_indices;
      std::vector<types::global_dof_index> p_local_dof_indices;

      MappingData(const unsigned int v_dofs_per_cell,
                  const unsigned int p_dofs_per_cell);
    };
    template <int dim>
    struct LocalCellData
    {
      FEValues<dim>         v_fe_values;
      FEValues<dim>         p_fe_values;
      unsigned int          n_q_points;
      unsigned int          v_dofs_per_cell;
      unsigned int          p_dofs_per_cell;
      std::vector<double>   div_phi_v;
      std::vector<double>   phi_p;
      LocalCellData(const FESystem<dim> &v_fe,
                    const FE_Q<dim>     &p_fe,
                    const QGauss<dim>   &quadrature_formula,
                    const UpdateFlags   v_flags,
                    const UpdateFlags   p_flags);
      LocalCellData(const LocalCellData &data);
    };
  }// namespace PressureGradientTermAssembly
} // namespace Step35

#endif /* INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_ */