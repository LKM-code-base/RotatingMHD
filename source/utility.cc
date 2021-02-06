#include <rotatingMHD/utility.h>

namespace RMHD
{

using namespace RunTimeParameters;

template <typename MatrixType>
void build_preconditioner
(std::shared_ptr<LinearAlgebra::PreconditionBase>     &preconditioner,
 const MatrixType &matrix,
 const PreconditionBaseParameters  *parameters,
 const bool higher_order_elements,
 const bool symmetric)
{
  preconditioner.reset();

  switch (parameters->preconditioner_type)
  {
    case (PreconditionerType::ILU):
    {
      LinearAlgebra::MPI::PreconditionILU::AdditionalData preconditioner_data;

      const PreconditionILUParameters* preconditioner_parameters
        = static_cast<const PreconditionILUParameters*>(parameters);

      #ifdef USE_PETSC_LA
        preconditioner_data.levels = preconditioner_parameters->fill;
      #else
        preconditioner_data.ilu_fill = preconditioner_parameters->fill;
        preconditioner_data.overlap = preconditioner_parameters->overlap;
        preconditioner_data.ilu_rtol = preconditioner_parameters->relative_tolerance;
        preconditioner_data.ilu_atol = preconditioner_parameters->absolute_tolerance;
      #endif

      preconditioner =
          std::make_shared<LinearAlgebra::MPI::PreconditionILU>();

      static_cast<LinearAlgebra::MPI::PreconditionILU*>(preconditioner.get())
          ->initialize(matrix, preconditioner_data);

      break;
    }
    case (PreconditionerType::AMG):
    {
      LinearAlgebra::MPI::PreconditionAMG::AdditionalData preconditioner_data;

      const PreconditionAMGParameters* preconditioner_parameters
        = static_cast<const PreconditionAMGParameters*>(parameters);

      #ifdef USE_PETSC_LA
        preconditioner_data.symmetric_operator = symmetric;
        preconditioner_data.strong_threshold = preconditioner_parameters->strong_threshold;
      #else
        preconditioner_data.elliptic = preconditioner_parameters->elliptic;
        preconditioner_data.higher_order_elements = higher_order_elements;
        preconditioner_data.n_cycles = preconditioner_parameters->n_cycles;
        preconditioner_data.aggregation_threshold = preconditioner_parameters->aggregation_threshold;
      #endif

        preconditioner =
          std::make_shared<LinearAlgebra::MPI::PreconditionILU>();

        static_cast<LinearAlgebra::MPI::PreconditionAMG*>(preconditioner.get())
            ->initialize(matrix,
                         preconditioner_data);
      break;
    }
    case (PreconditionerType::SOR):
    {
      LinearAlgebra::MPI::PreconditionSSOR::AdditionalData preconditioner_data;

      preconditioner_data.omega = 1.2;

      preconditioner =
          std::make_shared<LinearAlgebra::MPI::PreconditionSSOR>();

      static_cast<LinearAlgebra::MPI::PreconditionSSOR*>(preconditioner.get())
          ->initialize(matrix,
                       preconditioner_data);
      break;
    }
    case PreconditionerType::Jacobi:
    {
      LinearAlgebra::MPI::PreconditionJacobi::AdditionalData preconditioner_data;

      preconditioner =
          std::make_shared<LinearAlgebra::MPI::PreconditionJacobi>();

      static_cast<LinearAlgebra::MPI::PreconditionJacobi*>(preconditioner.get())
          ->initialize(matrix,
                       preconditioner_data);
      break;
    }
    case RunTimeParameters::PreconditionerType::GMG:
    {
      Assert(false, ExcNotImplemented());
      break;
    }
    default:
    {
      Assert(false,
             ExcMessage("The specified type of the preconditioner is unknown."));
    }
  }
}

}  // namespace RMD

// explicit instantiations
template void RMHD::build_preconditioner
(std::shared_ptr<LinearAlgebra::PreconditionBase> &,
 const RMHD::LinearAlgebra::MPI::SparseMatrix &matrix,
 const RunTimeParameters::PreconditionBaseParameters  *,
 const bool ,
 const bool );
