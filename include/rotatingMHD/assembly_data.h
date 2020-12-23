#ifndef INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_
#define INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/numerics/matrix_tools.h>

namespace RMHD
{
using namespace dealii;

namespace AssemblyData
{

template <int dim>  
struct CopyBase
{
  CopyBase(const unsigned int dofs_per_cell);

  CopyBase(const CopyBase &data);

  unsigned int                          dofs_per_cell;

  std::vector<types::global_cell_index> local_dof_indices;
};

template <int dim>  
struct ScratchBase
{
  ScratchBase(const Quadrature<dim>     &quadrature_formula,
              const FiniteElement<dim>  &fe);

  ScratchBase(const ScratchBase &data);

  const unsigned int  n_q_points;

  const unsigned int  dofs_per_cell;
};

namespace Generic
{

namespace Matrix
{

template <int dim>  
struct Copy : CopyBase<dim>
{
  Copy(const unsigned int dofs_per_cell);

  Copy(const Copy &data);

  FullMatrix<double>  local_matrix;
};

template <int dim>  
struct MassStiffnessCopy : CopyBase<dim>
{
  MassStiffnessCopy(const unsigned int dofs_per_cell);

  MassStiffnessCopy(const MassStiffnessCopy &data);

  FullMatrix<double>  local_mass_matrix;

  FullMatrix<double>  local_stiffness_matrix;
};

template <int dim>  
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch &data);

  FEValues<dim> fe_values;
};

} // namespace Matrix
 
namespace RightHandSide
{

template <int dim>  
struct Copy : CopyBase<dim>
{
  Copy(const unsigned int dofs_per_cell);

  Copy(const Copy &data);

  Vector<double>      local_rhs;

  FullMatrix<double>  local_matrix_for_inhomogeneous_bc;
};

template <int dim>  
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch &data);

  FEValues<dim> fe_values;
};

} // namespace RightHandSide

} // namespace Generic

namespace NavierStokesProjection
{

namespace VelocityConstantMatrices
{
template<int dim>
using Copy = Generic::Matrix::MassStiffnessCopy<dim>;

template <int dim>  
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);
  
  Scratch(const Scratch &data);

  std::vector<Tensor<1, dim>> phi;
  
  std::vector<Tensor<2, dim>> grad_phi;
};

} // namespace VelocityConstantMatrices

namespace PressureConstantMatrices
{

template<int dim>
using Copy = Generic::Matrix::MassStiffnessCopy<dim>;

template <int dim>  
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);
  
  Scratch(const Scratch &data);

  std::vector<double>         phi;
  
  std::vector<Tensor<1, dim>> grad_phi;
};

} // namespace PressureConstantMatrices

namespace AdvectionMatrix
{

} // namespace AdvectionMatrix

namespace DiffusionStepRHS
{

} // DiffusionStepRHS

namespace ProjectionStepRHS
{

} // ProjectionStepRHS

} // namespace NavierStokesProjection

namespace HeatEquation
{

namespace ConstantMatrices
{

template<int dim>
using Copy = Generic::Matrix::MassStiffnessCopy<dim>;

template <int dim>  
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);
  
  Scratch(const Scratch &data);

  std::vector<double>         phi;
  
  std::vector<Tensor<1, dim>> grad_phi;
};

} // namespace ConstantMatrices

namespace AdvectionMatrix
{

template<int dim>
using Copy = Generic::Matrix::Copy<dim>;

template <int dim>  
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &temperature_fe,
          const UpdateFlags         temperature_update_flags,
          const FiniteElement<dim>  &velocity_fe,
          const UpdateFlags         velocity_update_flags);
  
  Scratch(const Scratch &data);

  FEValues<dim>               temperature_fe_values;

  FEValues<dim>               velocity_fe_values;
  
  std::vector<Tensor<1,dim>>  velocity_values;
  
  std::vector<Tensor<1,dim>>  old_velocity_values;

  std::vector<Tensor<1,dim>>  old_old_velocity_values;

  std::vector<double>         phi;
  
  std::vector<Tensor<1,dim>>  grad_phi;
};

} // namespace AdvectionMatrix

namespace RightHandSide
{

template<int dim>
using Copy = Generic::RightHandSide::Copy<dim>;

template <int dim>  
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const Quadrature<dim-1>   &face_quadrature_formula,
          const FiniteElement<dim>  &temperature_fe,
          const UpdateFlags         temperature_update_flags,
          const UpdateFlags         temperature_face_update_flags,
          const FiniteElement<dim>  &velocity_fe,
          const UpdateFlags         velocity_update_flags);
  
  Scratch(const Scratch &data);

  FEValues<dim>               temperature_fe_values;

  FEFaceValues<dim>           temperature_fe_face_values;

  FEValues<dim>               velocity_fe_values;

  const unsigned int          n_face_q_points;

  std::vector<double>         alpha;

  std::vector<double>         temperature_values;

  std::vector<double>         old_temperature_values;
  
  std::vector<double>         old_old_temperature_values;

  std::vector<double>         beta;

  std::vector<Tensor<1,dim>>  velocity_values;

  std::vector<Tensor<1,dim>>  old_velocity_values;

  std::vector<Tensor<1,dim>>  old_old_velocity_values;

  std::vector<Tensor<1,dim>>  temperature_gradients;

  std::vector<Tensor<1,dim>>  old_temperature_gradients;

  std::vector<Tensor<1,dim>>  old_old_temperature_gradients;

  std::vector<double>         gamma;

  std::vector<double>         source_term_values;

  std::vector<double>         old_source_term_values;

  std::vector<double>         old_old_source_term_values;

  std::vector<double>         neuamnn_bc_values;

  std::vector<double>         old_neuamnn_bc_values;

  std::vector<double>         old_old_neuamnn_bc_values;

  std::vector<double>         phi;
  
  std::vector<Tensor<1,dim>>  grad_phi;

  std::vector<double>         face_phi;
};

} // namespace RightHandSide

} // namespace HeatEquation

} // namespace AssemblyData

namespace AdvectionAssembly
{

  template <int dim>
struct MappingData
{
  FullMatrix<double>                   local_matrix;
  std::vector<types::global_dof_index> local_dof_indices;

  MappingData(const unsigned int dofs_per_cell);
  MappingData(const MappingData &data);
};

template <int dim>  
struct LocalCellData
{
  using CurlType = typename FEValuesViews::Vector< dim >::curl_type;

  FEValues<dim>               fe_values;
  const unsigned int          n_q_points;
  const unsigned int          dofs_per_cell;
  std::vector<double>         extrapolated_velocity_divergences;
  std::vector<Tensor<1,dim>>  extrapolated_velocity_values;
  std::vector<CurlType>       extrapolated_velocity_curls;
  std::vector<Tensor<1,dim>>  phi_velocity;
  std::vector<Tensor<2,dim>>  grad_phi_velocity;
  std::vector<CurlType>       curl_phi_velocity;

  LocalCellData(const FESystem<dim>  &fe,
                const Quadrature<dim>&quadrature_formula,
                const UpdateFlags     flags);
  LocalCellData(const LocalCellData  &data);
};
}// namespace AdvectionAssembly

namespace VelocityRightHandSideAssembly
{
template <int dim>
struct MappingData
{
  unsigned int                          velocity_dofs_per_cell;
  Vector<double>                        local_diffusion_step_rhs;
  FullMatrix<double>                    local_matrix_for_inhomogeneous_bc;
  std::vector<types::global_dof_index>  local_velocity_dof_indices;

  MappingData(const unsigned int velocity_dofs_per_cell);
  MappingData(const MappingData &data);
};

template <int dim>
struct LocalCellData
{
  using CurlType = typename FEValuesViews::Vector< dim >::curl_type;

  FEValues<dim>                         velocity_fe_values;
  FEValues<dim>                         pressure_fe_values;
  FEValues<dim>                         temperature_fe_values;
  const unsigned int                    n_q_points;
  const unsigned int                    velocity_dofs_per_cell;
  std::vector<double>                   pressure_tmp_values;
  std::vector<Tensor<1, dim>>           velocity_tmp_values;
  std::vector<Tensor<1, dim>>           phi_velocity;
  std::vector<double>                   div_phi_velocity;
  std::vector<double>                   extrapolated_velocity_divergences;
  std::vector<Tensor<1,dim>>            extrapolated_velocity_values;
  std::vector<CurlType>                 extrapolated_velocity_curls;
  std::vector<double>                   old_velocity_divergences;
  std::vector<Tensor<1, dim>>           old_velocity_values;
  std::vector<CurlType>                 old_velocity_curls;
  std::vector<Tensor<2,dim>>            old_velocity_gradients;
  std::vector<double>                   old_old_velocity_divergences;
  std::vector<Tensor<1, dim>>           old_old_velocity_values;
  std::vector<CurlType>                 old_old_velocity_curls;
  std::vector<Tensor<2,dim>>            old_old_velocity_gradients;
  std::vector<double>                   extrapolated_temperature_values;
  std::vector<Tensor<1,dim>>            body_force_values;
  std::vector<Tensor<2,dim>>            grad_phi_velocity;
  std::vector<CurlType>                 curl_phi_velocity;

  LocalCellData(const FESystem<dim>  &velocity_fe,
                const FE_Q<dim>      &pressure_fe,
                const FE_Q<dim>      &temperature_fe,
                const Quadrature<dim>&velocity_quadrature_formula,
                const UpdateFlags     velocity_update_flags,
                const UpdateFlags     pressure_update_flags,
                const UpdateFlags     temperature_update_flags);
  LocalCellData(const LocalCellData  &data);
};
} // namespace VelocityRightHandSideAssembly

namespace PressureRightHandSideAssembly
{
template <int dim>
struct MappingData
{
  unsigned int                          pressure_dofs_per_cell;
  Vector<double>                        local_projection_step_rhs;
  FullMatrix<double>                    local_matrix_for_inhomogeneous_bc;
  std::vector<types::global_dof_index>  local_pressure_dof_indices;

  MappingData(const unsigned int pressure_dofs_per_cell);
  MappingData(const MappingData &data);
};

template <int dim>
struct LocalCellData
{
  FEValues<dim>                         velocity_fe_values;
  FEValues<dim>                         pressure_fe_values;
  const unsigned int                    n_q_points;
  const unsigned int                    pressure_dofs_per_cell;
  std::vector<double>                   velocity_divergence_values;
  std::vector<double>                   phi_pressure;
  std::vector<Tensor<1, dim>>           grad_phi_pressure;

  LocalCellData(const FESystem<dim>  &velocity_fe,
                const FE_Q<dim>      &pressure_fe,
                const Quadrature<dim>&pressure_quadrature_formula,
                const UpdateFlags     velocity_update_flags,
                const UpdateFlags     pressure_update_flags);
  LocalCellData(const LocalCellData  &data);
};
} // namespace PressureRightHandSideAssembly

namespace PoissonPrestepRightHandSideAssembly
{

template <int dim>
struct MappingData
{
  MappingData(const unsigned int pressure_dofs_per_cell);

  MappingData(const MappingData &data);

  unsigned int                          pressure_dofs_per_cell;

  Vector<double>                        local_poisson_prestep_rhs;

  FullMatrix<double>                    local_matrix_for_inhomogeneous_bc;

  std::vector<types::global_dof_index>  local_pressure_dof_indices;
};

template <int dim>
struct LocalCellData
{
  LocalCellData(const FESystem<dim>     &velocity_fe,
                const FE_Q<dim>         &pressure_fe,
                const FE_Q<dim>         &temperature_fe,
                const Quadrature<dim>   &pressure_quadrature_formula,
                const Quadrature<dim-1> &pressure_face_quadrature_formula,
                const UpdateFlags        velocity_face_update_flags,
                const UpdateFlags        pressure_update_flags,
                const UpdateFlags        pressure_face_update_flags,
                const UpdateFlags        temperature_update_flags,
                const UpdateFlags        temperature_face_update_flags);

  LocalCellData(const LocalCellData     &data);

  FEValues<dim>                         pressure_fe_values;

  FEValues<dim>                         temperature_fe_values;

  FEFaceValues<dim>                     velocity_fe_face_values;

  FEFaceValues<dim>                     pressure_fe_face_values;

  FEFaceValues<dim>                     temperature_fe_face_values;

  const unsigned int                    n_q_points;
  const unsigned int                    n_face_q_points;
  const unsigned int                    pressure_dofs_per_cell;

  std::vector<double>                   body_force_divergence_values;
  
  std::vector<Tensor<1, dim>>           temperature_gradient_values;

  std::vector<Tensor<1, dim>>           velocity_laplacian_values;

  std::vector<Tensor<1, dim>>           body_force_values;

  std::vector<double>                   temperature_values;

  std::vector<Tensor<1, dim>>           normal_vectors;

  std::vector<double>                   phi_pressure;
  std::vector<double>                   face_phi_pressure;
  std::vector<Tensor<1, dim>>           grad_phi_pressure;
};

}


namespace TemperatureConstantMatricesAssembly
{

template <int dim>
struct MappingData
{
  
  unsigned int                          dofs_per_cell;
  
  FullMatrix<double>                    local_mass_matrix;

  FullMatrix<double>                    local_stiffness_matrix;

  std::vector<types::global_dof_index>  local_dof_indices;

  MappingData(const unsigned int dofs_per_cell);
  
  MappingData(const MappingData &data);
};

template <int dim>
struct LocalCellData
{
  
  FEValues<dim>               fe_values;
  
  const unsigned int          n_q_points;
  
  const unsigned int          dofs_per_cell;
  
  std::vector<double>         phi;

  std::vector<Tensor<1,dim>>  grad_phi;

  LocalCellData(const Mapping<dim>   &mapping,
                const FE_Q<dim>      &fe,
                const Quadrature<dim>&quadrature_formula,
                const UpdateFlags     update_flags);
  
  LocalCellData(const LocalCellData  &data);
};

} // namespace TemperatureMassMatrixAssembly

namespace TemperatureAdvectionMatrixAssembly
{

template <int dim>
struct MappingData
{
  
  unsigned int                          dofs_per_cell;
  
  FullMatrix<double>                    local_matrix;
  
  std::vector<types::global_dof_index>  local_dof_indices;

  MappingData(const unsigned int dofs_per_cell);
  
  MappingData(const MappingData &data);
};

template <int dim>
struct LocalCellData
{
  FEValues<dim>               temperature_fe_values;

  FEValues<dim>               velocity_fe_values;

  const unsigned int          n_q_points;
  
  const unsigned int          dofs_per_cell;
  
  std::vector<Tensor<1,dim>>  velocity_values;
  
  std::vector<double>         phi;
  
  std::vector<Tensor<1,dim>>  grad_phi;

  LocalCellData(const Mapping<dim>   &mapping,
                const FE_Q<dim>      &temperature_fe,
                const FESystem<dim>  &velocity_fe,
                const Quadrature<dim>&quadrature_formula,
                const UpdateFlags     temperature_update_flags,
                const UpdateFlags     velocity_update_flags);
  
  LocalCellData(const LocalCellData  &data);
};

} // namespace TemperatureAdvectionMatrixAssembly

namespace TemperatureRightHandSideAssembly
{

template <int dim>
struct MappingData
{
  unsigned int                          dofs_per_cell;

  Vector<double>                        local_rhs;

  FullMatrix<double>                    local_matrix_for_inhomogeneous_bc;

  std::vector<types::global_dof_index>  local_dof_indices;

  MappingData(const unsigned int dofs_per_cell);

  MappingData(const MappingData &data);
};

template <int dim>
struct LocalCellData
{
  FEValues<dim>                         temperature_fe_values;

  FEValues<dim>                         velocity_fe_values;

  FEFaceValues<dim>                     temperature_fe_face_values;

  const unsigned int                    n_q_points;

  const unsigned int                    n_face_q_points;

  const unsigned int                    dofs_per_cell;

  std::vector<double>                   temperature_tmp_values;

  std::vector<double>                   source_term_values;
  
  std::vector<double>                   neumann_bc_values;

  std::vector<double>                   old_temperature_values;
  
  std::vector<double>                   old_old_temperature_values;

  std::vector<Tensor<1,dim>>            old_temperature_gradients;

  std::vector<Tensor<1,dim>>            old_old_temperature_gradients;

  std::vector<Tensor<1,dim>>            velocity_values;

  std::vector<double>                   phi;

  std::vector<Tensor<1,dim>>            grad_phi;

  std::vector<double>                   face_phi;

  LocalCellData(const Mapping<dim>      &mapping,
                const FE_Q<dim>         &temperature_fe,
                const FESystem<dim>     &velocity_fe,
                const Quadrature<dim>   &quadrature_formula,
                const Quadrature<dim-1> &face_quadrature_formula,
                const UpdateFlags        temperature_update_flags,
                const UpdateFlags        velocity_update_flags,
                const UpdateFlags        temperature_face_update_flags);

  LocalCellData(const LocalCellData     &data);
};

} // namespace TemperatureRightHandSideAssembly

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_ */
