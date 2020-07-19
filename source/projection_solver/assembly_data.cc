/*
 * assembly_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/projection_solver.h>

namespace Step35
{

template<int dim>
NavierStokesProjection<dim>::InitGradPerTaskData::InitGradPerTaskData
(const unsigned int vdpc,
 const unsigned int pdpc)
:
vel_dpc(vdpc),
pres_dpc(pdpc),
local_grad(vdpc, pdpc),
vel_local_dof_indices(vdpc),
pres_local_dof_indices(pdpc)
{}

template<int dim>
NavierStokesProjection<dim>::InitGradScratchData::InitGradScratchData
(const FESystem<dim> &  fe_v,
 const FE_Q<dim> &  fe_p,
 const QGauss<dim> &quad,
 const UpdateFlags  flags_v,
 const UpdateFlags  flags_p)
:
nqp(quad.size()),
fe_val_vel(fe_v, quad, flags_v),
fe_val_pres(fe_p, quad, flags_p)
{}

template<int dim>
NavierStokesProjection<dim>::InitGradScratchData::InitGradScratchData
(const InitGradScratchData &data)
:
nqp(data.nqp),
fe_val_vel(data.fe_val_vel.get_fe(),
           data.fe_val_vel.get_quadrature(),
           data.fe_val_vel.get_update_flags()),
fe_val_pres(data.fe_val_pres.get_fe(),
            data.fe_val_pres.get_quadrature(),
            data.fe_val_pres.get_update_flags())
{}

template<int dim>
NavierStokesProjection<dim>::AdvectionPerTaskData::AdvectionPerTaskData(const unsigned int dpc)
:
local_advection(dpc, dpc),
local_dof_indices(dpc)
{}

template<int dim>
NavierStokesProjection<dim>::AdvectionScratchData::AdvectionScratchData
(const FESystem<dim> &  fe,
 const QGauss<dim> &quad,
 const UpdateFlags  flags)
:
nqp(quad.size()),
dpc(fe.dofs_per_cell),
div_u_star(nqp),
u_star_local(nqp),
fe_val(fe, quad, flags)
{}

template<int dim>
NavierStokesProjection<dim>::AdvectionScratchData::AdvectionScratchData
(const AdvectionScratchData &data)
:
nqp(data.nqp),
dpc(data.dpc),
div_u_star(nqp),
u_star_local(nqp),
fe_val(data.fe_val.get_fe(),
       data.fe_val.get_quadrature(),
       data.fe_val.get_update_flags())
{}

}  // namespace Step35

// explicit instantiations

template struct Step35::NavierStokesProjection<2>::InitGradPerTaskData;
template struct Step35::NavierStokesProjection<3>::InitGradPerTaskData;

template struct Step35::NavierStokesProjection<2>::InitGradScratchData;
template struct Step35::NavierStokesProjection<3>::InitGradScratchData;

template struct Step35::NavierStokesProjection<2>::AdvectionPerTaskData;
template struct Step35::NavierStokesProjection<3>::AdvectionPerTaskData;

template struct Step35::NavierStokesProjection<2>::AdvectionScratchData;
template struct Step35::NavierStokesProjection<3>::AdvectionScratchData;
