/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Abner Salgado, Texas A&M University 2009
 */

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/thread_management.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/sparse_ilu.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <fstream>
#include <cmath>
#include <iostream>

namespace Step35
{
  using namespace dealii;
  
  namespace RunTimeParameters
  {
    enum class Method
    {
      standard,
      rotational
    };

    class Data_Storage
    {
    public:
      Data_Storage();

      void read_data(const std::string &filename);

      Method form;

      double dt;
      double initial_time;
      double final_time;

      double Reynolds;

      unsigned int n_global_refines;

      unsigned int pressure_degree;

      unsigned int vel_max_iterations;
      unsigned int vel_Krylov_size;
      unsigned int vel_off_diagonals;
      unsigned int vel_update_prec;
      double       vel_eps;
      double       vel_diag_strength;

      bool         verbose;
      unsigned int output_interval;

    protected:
      ParameterHandler prm;
    };

    Data_Storage::Data_Storage()
      : form(Method::rotational)
      , dt(5e-4)
      , initial_time(0.)
      , final_time(1.)
      , Reynolds(1.)
      , n_global_refines(0)
      , pressure_degree(1)
      , vel_max_iterations(1000)
      , vel_Krylov_size(30)
      , vel_off_diagonals(60)
      , vel_update_prec(15)
      , vel_eps(1e-12)
      , vel_diag_strength(0.01)
      , verbose(true)
      , output_interval(15)
    {
      prm.declare_entry("Method_Form",
                        "rotational",
                        Patterns::Selection("rotational|standard"),
                        " Used to select the type of method that we are going "
                        "to use. ");
      prm.enter_subsection("Physical data");
      {
        prm.declare_entry("initial_time",
                          "0.",
                          Patterns::Double(0.),
                          " The initial time of the simulation. ");
        prm.declare_entry("final_time",
                          "1.",
                          Patterns::Double(0.),
                          " The final time of the simulation. ");
        prm.declare_entry("Reynolds",
                          "1.",
                          Patterns::Double(0.),
                          " The Reynolds number. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        prm.declare_entry("dt",
                          "5e-4",
                          Patterns::Double(0.),
                          " The time step size. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        prm.declare_entry("n_of_refines",
                          "0",
                          Patterns::Integer(0, 15),
                          " The number of global refines we do on the mesh. ");
        prm.declare_entry("pressure_fe_degree",
                          "1",
                          Patterns::Integer(1, 5),
                          " The polynomial degree for the pressure space. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        prm.declare_entry(
          "max_iterations",
          "1000",
          Patterns::Integer(1, 1000),
          " The maximal number of iterations GMRES must make. ");
        prm.declare_entry("eps",
                          "1e-12",
                          Patterns::Double(0.),
                          " The stopping criterion. ");
        prm.declare_entry("Krylov_size",
                          "30",
                          Patterns::Integer(1),
                          " The size of the Krylov subspace to be used. ");
        prm.declare_entry("off_diagonals",
                          "60",
                          Patterns::Integer(0),
                          " The number of off-diagonal elements ILU must "
                          "compute. ");
        prm.declare_entry("diag_strength",
                          "0.01",
                          Patterns::Double(0.),
                          " Diagonal strengthening coefficient. ");
        prm.declare_entry("update_prec",
                          "15",
                          Patterns::Integer(1),
                          " This number indicates how often we need to "
                          "update the preconditioner");
      }
      prm.leave_subsection();

      prm.declare_entry("verbose",
                        "true",
                        Patterns::Bool(),
                        " This indicates whether the output of the solution "
                        "process should be verbose. ");

      prm.declare_entry("output_interval",
                        "1",
                        Patterns::Integer(1),
                        " This indicates between how many time steps we print "
                        "the solution. ");
    }



    void Data_Storage::read_data(const std::string &filename)
    {
      std::ifstream file(filename);
      AssertThrow(file, ExcFileNotOpen(filename));

      prm.parse_input(file);

      if (prm.get("Method_Form") == std::string("rotational"))
        form = Method::rotational;
      else
        form = Method::standard;

      prm.enter_subsection("Physical data");
      {
        initial_time = prm.get_double("initial_time");
        final_time   = prm.get_double("final_time");
        Reynolds     = prm.get_double("Reynolds");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time step data");
      {
        dt = prm.get_double("dt");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization");
      {
        n_global_refines = prm.get_integer("n_of_refines");
        pressure_degree  = prm.get_integer("pressure_fe_degree");
      }
      prm.leave_subsection();

      prm.enter_subsection("Data solve velocity");
      {
        vel_max_iterations = prm.get_integer("max_iterations");
        vel_eps            = prm.get_double("eps");
        vel_Krylov_size    = prm.get_integer("Krylov_size");
        vel_off_diagonals  = prm.get_integer("off_diagonals");
        vel_diag_strength  = prm.get_double("diag_strength");
        vel_update_prec    = prm.get_integer("update_prec");
      }
      prm.leave_subsection();

      verbose = prm.get_bool("verbose");

      output_interval = prm.get_integer("output_interval");
    }
  } // namespace RunTimeParameters

  namespace EquationData
  {
    template <int dim>
    class VelocityInitialConditions : public Function<dim>
    {
    public:
      VelocityInitialConditions(const double initial_time = 0)
        : Function<dim>(dim, initial_time)
      {}
      virtual void vector_value(const Point<dim>  &p,
                                Vector<double>    &values) const override;
    };
    
    template <int dim>
    void VelocityInitialConditions<dim>::vector_value(
                                        const Point<dim>  &p,
                                        Vector<double>    &values) const
    {
      values[0] = 0.0;
      values[1] = 0.0;
    }

    template <int dim>
    class VelocityBC : public Function<dim>
    {
    public:
      VelocityBC(const double initial_time = 0)
        : Function<dim>(dim, initial_time)
      {}
      virtual void vector_value(const Point<dim>  &p,
                                Vector<double>    &values) const override;
    };
    
    template <int dim>
    void VelocityBC<dim>::vector_value(
                                        const Point<dim>  &p,
                                        Vector<double>    &values) const
    {
      const double Um = 1.5;
      const double H  = 4.1;

      values[0] = 4.0 * Um * p(1) * ( H * p(1) ) / ( H * H );
      values[1] = 0.0;
    }

    template <int dim>
    class PressureInitialCondition : public Function<dim>
    {
    public:
      PressureInitialCondition(const double initial_time = 0)
        : Function<dim>(1, initial_time)
      {}
      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const override;
    };

    template<int dim>
    double PressureInitialCondition<dim>::value(const Point<dim> &p,
                                           const unsigned int component) const
    {
      (void)component;
      return 25.0 - p(0);
    }
  } // namespace EquationData

  template <int dim>
  class NavierStokesProjection
  {
  public:
    NavierStokesProjection(const RunTimeParameters::Data_Storage &data);
    void run(const bool verbose = false, const unsigned int n_plots = 10);
  
  protected:
    RunTimeParameters::Method                  type;
    const unsigned int                        deg;
    const double                              dt;
    const double                              t_0;
    const double                              T;
    const double                              Re;

    EquationData::VelocityBC<dim>             vel_exact;
    std::map<types::global_dof_index, double> boundary_values;
    std::vector<types::boundary_id>            boundary_ids;

    Triangulation<dim>                        triangulation;
    FESystem<dim>                             fe_velocity;
    FE_Q<dim>                                 fe_pressure;

    DoFHandler<dim>                           dof_handler_velocity;
    DoFHandler<dim>                           dof_handler_pressure;
    QGauss<dim>                               quadrature_velocity;
    QGauss<dim>                               quadrature_pressure;

    SparsityPattern                           sparsity_pattern_velocity;
    SparsityPattern                           sparsity_pattern_pressure;
    SparsityPattern                           sparsity_pattern_pres_vel;

    SparseMatrix<double>                      vel_Laplace_plus_Mass;
    SparseMatrix<double>                      vel_it_matrix;
    SparseMatrix<double>                      vel_Mass;
    SparseMatrix<double>                      vel_Laplace;
    SparseMatrix<double>                      vel_Advection;
    SparseMatrix<double>                      pres_Laplace;
    SparseMatrix<double>                      pres_Mass;
    SparseMatrix<double>                      pres_Diff;
    SparseMatrix<double>                      pres_iterative;
  
    Vector<double>                            pres_n;
    Vector<double>                            pres_n_minus_1;
    Vector<double>                            phi_n;
    Vector<double>                            phi_n_minus_1;
    Vector<double>                            u_n;
    Vector<double>                            u_n_minus_1;
    Vector<double>                            u_star;
    Vector<double>                            force;
    Vector<double>                            v_tmp;
    Vector<double>                            pres_tmp;
    Vector<double>                            rot_u;

    SparseILU<double>                         prec_velocity;
    SparseILU<double>                         prec_pres_Laplace;
    SparseDirectUMFPACK                       prec_mass;
    SparseDirectUMFPACK                       prec_vel_mass;

    DeclException2(ExcInvalidTimeStep,
                  double,
                  double,
                  << " The time step " << arg1 << " is out of range."
                  << std::endl
                  << " The permitted range is (0," << arg2 << "]");

    void create_triangulation_and_dofs(const unsigned int n_refines);
    void initialize();
    void interpolate_velocity();
    void diffusion_step(const bool reinit_prec);
    void projection_step(const bool reinit_prec);
    void update_pressure(const bool reinit_prec);

  private:
    unsigned int                              vel_max_its;
    unsigned int                              vel_Krylov_size;
    unsigned int                              vel_off_diagonals;
    unsigned int                              vel_update_prec;
    double                                    vel_eps;
    double                                    vel_diag_strength;

    struct InitGradPerTaskData
    {
      unsigned int                         d;
      unsigned int                         vel_dpc;
      unsigned int                         pres_dpc;
      FullMatrix<double>                   local_grad;
      std::vector<types::global_dof_index> vel_local_dof_indices;
      std::vector<types::global_dof_index> pres_local_dof_indices;
      InitGradPerTaskData(const unsigned int dd,
                          const unsigned int vdpc,
                          const unsigned int pdpc)
        : d(dd)
        , vel_dpc(vdpc)
        , pres_dpc(pdpc)
        , local_grad(vdpc, pdpc)
        , vel_local_dof_indices(vdpc)
        , pres_local_dof_indices(pdpc)
      {}
    };
    struct InitGradScratchData
    {
      unsigned int  nqp;
      FEValues<dim> fe_val_vel;
      FEValues<dim> fe_val_pres;
      InitGradScratchData(const FE_Q<dim> &  fe_v,
                          const FE_Q<dim> &  fe_p,
                          const QGauss<dim> &quad,
                          const UpdateFlags  flags_v,
                          const UpdateFlags  flags_p)
        : nqp(quad.size())
        , fe_val_vel(fe_v, quad, flags_v)
        , fe_val_pres(fe_p, quad, flags_p)
      {}
      InitGradScratchData(const InitGradScratchData &data)
        : nqp(data.nqp)
        , fe_val_vel(data.fe_val_vel.get_fe(),
                    data.fe_val_vel.get_quadrature(),
                    data.fe_val_vel.get_update_flags())
        , fe_val_pres(data.fe_val_pres.get_fe(),
                      data.fe_val_pres.get_quadrature(),
                      data.fe_val_pres.get_update_flags())
      {}
    };

    struct AdvectionPerTaskData
    {
      FullMatrix<double>                   local_advection;
      std::vector<types::global_dof_index> local_dof_indices;
      AdvectionPerTaskData(const unsigned int dpc)
        : local_advection(dpc, dpc)
        , local_dof_indices(dpc)
      {}
    };
    struct AdvectionScratchData
    {
      unsigned int                nqp;
      unsigned int                dpc;
      std::vector<Point<dim>>     u_star_local;
      std::vector<Tensor<1, dim>> grad_u_star;
      std::vector<double>         u_star_tmp;
      FEValues<dim>               fe_val;
      AdvectionScratchData(const FE_Q<dim> &  fe,
                          const QGauss<dim> &quad,
                          const UpdateFlags  flags)
        : nqp(quad.size())
        , dpc(fe.dofs_per_cell)
        , u_star_local(nqp)
        , grad_u_star(nqp)
        , u_star_tmp(nqp)
        , fe_val(fe, quad, flags)
      {}
      AdvectionScratchData(const AdvectionScratchData &data)
        : nqp(data.nqp)
        , dpc(data.dpc)
        , u_star_local(nqp)
        , grad_u_star(nqp)
        , u_star_tmp(nqp)
        , fe_val(data.fe_val.get_fe(),
                data.fe_val.get_quadrature(),
                data.fe_val.get_update_flags())
      {}
    };

    using IteratorTuple = std::tuple<
                        typename DoFHandler<dim>::active_cell_iterator,
                        typename DoFHandler<dim>::active_cell_iterator>;
    using IteratorPair  = SynchronousIterators<IteratorTuple>;

    void initialize_velocity_matrices();
    void initialize_pressure_matrices();
    void initialize_gradient_operator();
    void assemble_one_cell_of_gradient(const IteratorPair   &SI,
                                       InitGradScratchData  &scratch,
                                       InitGradPerTaskData  &data);
    void copy_gradient_local_to_global(const InitGradPerTaskData &data);
    void assemble_advection_term();
    void assemble_one_cell_of_advection(
        const typename DoFHandler<dim>::active_cell_iterator  &cell,
        AdvectionScratchData                                  scratch,
        AdvectionPerTaskData                                  data);
    void copy_advection_local_to_global(const AdvectionPerTaskData &data);
    void diffusion_component_solve(const unsigned int d);
    void output_results(const unsigned int step);
    void assemble_vorticity(const bool reinit_prec);
  };

  template <int dim>
  NavierStokesProjection<dim>::NavierStokesProjection(
    const RunTimeParameters::Data_Storage &data)
    : type(data.form)
    , deg(data.pressure_degree)
    , dt(data.dt)
    , t_0(data.initial_time)
    , T(data.final_time)
    , Re(data.Reynolds)
    , vel_exact(data.initial_time)
    , fe_velocity(FE_Q<dim>(deg + 1), dim)
    , fe_pressure(deg)
    , dof_handler_velocity(triangulation)
    , dof_handler_pressure(triangulation)
    , quadrature_pressure(deg + 1)
    , quadrature_velocity(deg + 2)
    , vel_max_its(data.vel_max_iterations)
    , vel_Krylov_size(data.vel_Krylov_size)
    , vel_off_diagonals(data.vel_off_diagonals)
    , vel_update_prec(data.vel_update_prec)
    , vel_eps(data.vel_eps)
    , vel_diag_strength(data.vel_diag_strength)
  {
    if (deg < 1)
      std::cout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;
    AssertThrow(!((dt <= 0.) || (dt > .5 * T)), ExcInvalidTimeStep(dt, .5 * T));
    create_triangulation_and_dofs(data.n_global_refines);
    //initialize();
  }

  template <int dim>
  void NavierStokesProjection<dim>::create_triangulation_and_dofs(
    const unsigned int n_refines)
  {
    GridIn<dim> grid_in;
    grid_in.attach_triangulation(triangulation);
    {
      std::string   filename = "nsbench2.inp";
      std::ifstream file(filename);
      Assert(file, ExcFileNotOpen(filename.c_str()));
      grid_in.read_ucd(file);
    }
    std::cout << "Number of refines = " << n_refines << std::endl;
    triangulation.refine_global(n_refines);
    std::cout << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl;
    boundary_ids = triangulation.get_boundary_ids();
    dof_handler_velocity.distribute_dofs(fe_velocity);
    DoFRenumbering::boost::Cuthill_McKee(dof_handler_velocity);
    dof_handler_pressure.distribute_dofs(fe_pressure);
    DoFRenumbering::boost::Cuthill_McKee(dof_handler_pressure);
    //initialize_velocity_matrices();
    //initialize_pressure_matrices();
    //initialize_gradient_operator();
    pres_n.reinit(dof_handler_pressure.n_dofs());
    pres_n_minus_1.reinit(dof_handler_pressure.n_dofs());
    phi_n.reinit(dof_handler_pressure.n_dofs());
    phi_n_minus_1.reinit(dof_handler_pressure.n_dofs());
    pres_tmp.reinit(dof_handler_pressure.n_dofs());
    u_n.reinit(dof_handler_velocity.n_dofs());
    u_n_minus_1.reinit(dof_handler_velocity.n_dofs());
    u_star.reinit(dof_handler_velocity.n_dofs());
    force.reinit(dof_handler_velocity.n_dofs());
    v_tmp.reinit(dof_handler_velocity.n_dofs());
    rot_u.reinit(dof_handler_velocity.n_dofs());
    std::cout << "dim (X_h) = " << dof_handler_velocity.n_dofs() 
              << std::endl                                               
              << "dim (M_h) = " << dof_handler_pressure.n_dofs()         
              << std::endl                                               
              << "Re        = " << Re << std::endl                       
              << std::endl;
  }

} // namespace Step35

int main()
{
  try
    {
      using namespace dealii;
      using namespace Step35;

      RunTimeParameters::Data_Storage data;
      data.read_data("parameter-file.prm");
      //deallog.depth_console(data.verbose ? 2 : 0);

      NavierStokesProjection<2> test(data);
      //test.run(data.verbose, data.output_interval);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  std::cout << "----------------------------------------------------"
            << std::endl
            << "Apparently everything went fine!" << std::endl
            << "Don't forget to brush your teeth :-)" << std::endl
            << std::endl;
  return 0;
}