#include <rotatingMHD/magnetic_induction.h>



namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::assemble_zeroth_step_rhs()
{

}



} // namespace Solvers



} // namespace RMHD

// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::assemble_zeroth_step_rhs();
template void RMHD::Solvers::MagneticInduction<3>::assemble_zeroth_step_rhs();
