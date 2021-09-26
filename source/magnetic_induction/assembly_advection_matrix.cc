#include <rotatingMHD/magnetic_induction.h>



namespace RMHD
{



namespace Solvers
{



template <int dim>
void MagneticInduction<dim>::assemble_advection_matrix()
{

}



} // namespace Solvers



} // namespace RMHD

// Explicit instantiations
template void RMHD::Solvers::MagneticInduction<2>::assemble_advection_matrix();
template void RMHD::Solvers::MagneticInduction<3>::assemble_advection_matrix();
