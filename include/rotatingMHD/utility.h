/*
 * utility.h
 *
 *  Created on: Feb 5, 2021
 *      Author: sg
 */

#ifndef INCLUDE_ROTATINGMHD_UTILITY_H_
#define INCLUDE_ROTATINGMHD_UTILITY_H_

#include <rotatingMHD/global.h>
#include <rotatingMHD/run_time_parameters.h>

#include <memory>

namespace RMHD
{

namespace Utility
{

/*!
 * @brief This method sets up the #preconditioner of the System #matrix based
 * on the type and parameters specified in the #parameters.
 * This method is called inside NavierStokesProjection::solve_projection_step(),
 * NavierStokesProjection::solve_diffusion_step(),
 * NavierStokesProjection::solve_projection_step(),
 * NavierStokesProjection::solve_poisson_prestep() and
 * NavierStokesProjection::pressure_correction().
 */
template<typename MatrixType>
void build_preconditioner
(const std::shared_ptr<LinearAlgebra::PreconditionBase> &preconditioner,
 const MatrixType &matrix,
 const RunTimeParameters::PreconditionBaseParameters    *parameters,
 const bool higher_order_elements = false,
 const bool symmetric = true);

}  // namespace Utility

}  // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_UTILITY_H_ */
