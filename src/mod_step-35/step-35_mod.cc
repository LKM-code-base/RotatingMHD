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
    enum class ProjectionMethod
    {
      standard,
      rotational
    };

    class ParameterSet
    {
    public:
      ProjectionMethod  projection_method;
      double            dt;
      double            t_0;
      double            T;
      double            Re;
      unsigned int      n_global_refinements;
      unsigned int      p_fe_degree;
      unsigned int      solver_max_iterations;
      unsigned int      solver_krylov_size;
      unsigned int      solver_off_diagonals;
      unsigned int      solver_update_preconditioner;
      double            solver_tolerance;
      double            solver_diag_strength;
      bool              flag_verbose_output;
      unsigned int      output_interval;
  
      ParameterSet();
      void read_data_from_file(const std::string &filename);
    
    protected:
      ParameterHandler  prm;
    };

    ParameterSet::ParameterSet()
      : projection_method(ProjectionMethod::rotational),
        dt(5e-4),
        t_0(0.0),
        T(1.0),
        Re(1.0),
        n_global_refinements(0),
        p_fe_degree(1),
        solver_max_iterations(1000),
        solver_krylov_size(30),
        solver_off_diagonals(60),
        solver_update_preconditioner(15),
        solver_tolerance(1e-12),
        solver_diag_strength(0.01),
        flag_verbose_output(true),
        output_interval(15)
    {
      prm.declare_entry("projection_method",
                        "rotational",
                        Patterns::Selection("rotational|standard"),
                        " Projection method to implement. ");
      prm.enter_subsection("Physical parameters");
      {
        prm.declare_entry("Re",
                          "1.",
                          Patterns::Double(0.),
                          " Reynolds number. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time discretization parameters");
      {
        prm.declare_entry("dt",
                          "5e-4",
                          Patterns::Double(0.),
                          " Time step size. ");
        prm.declare_entry("t_0",
                          "0.",
                          Patterns::Double(0.),
                          " The initial time of the simulation. ");
        prm.declare_entry("T",
                          "1.",
                          Patterns::Double(0.),
                          " The final time of the simulation. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization parameters");
      {
        prm.declare_entry("n_global_refinements",
                          "0",
                          Patterns::Integer(0, 15),
                          " Number of global refinements done on the"
                            "input mesh. ");
        prm.declare_entry("p_fe_degree",
                          "1",
                          Patterns::Integer(1, 5),
                          " The polynomial degree of the pressure finite" 
                            "element. ");
      }
      prm.leave_subsection();

      prm.enter_subsection("Parameters of the diffusion-step solver");
      {
        prm.declare_entry("solver_max_iterations",
                          "1000",
                          Patterns::Integer(1, 1000),
                          " Maximal number of iterations done by the" 
                            "GMRES solver. ");
        prm.declare_entry("solver_tolerance",
                          "1e-12",
                          Patterns::Double(0.),
                          " Tolerance of the GMRES solver. ");
        prm.declare_entry("solver_krylov_size",
                          "30",
                          Patterns::Integer(1),
                          " The size of the Krylov subspace to be used. ");
        prm.declare_entry("solver_off_diagonals",
                          "60",
                          Patterns::Integer(0),
                          " The number of off-diagonal elements ILU must" 
                            "compute. ");
        prm.declare_entry("solver_diag_strength",
                          "0.01",
                          Patterns::Double(0.),
                          " Diagonal strengthening coefficient. ");
        prm.declare_entry("solver_update_preconditioner",
                          "15",
                          Patterns::Integer(1),
                          " This number indicates how often we need to" 
                            "update the preconditioner");
      }
      prm.leave_subsection();

      prm.declare_entry("flag_verbose_output",
                        "true",
                        Patterns::Bool(),
                        " This indicates whether the output of the" 
                          "solution process should be verbose. ");
      prm.declare_entry("output_interval",
                        "1",
                        Patterns::Integer(1),
                        " This indicates between how many time steps" 
                          "we print the solution. ");
    }

    void ParameterSet::
    read_data_from_file(const std::string &filename)
    {
      std::ifstream file(filename);
      AssertThrow(file, ExcFileNotOpen(filename));

      prm.parse_input(file);

      if (prm.get("projection_method") == std::string("rotational"))
        projection_method = ProjectionMethod::rotational;
      else
        projection_method = ProjectionMethod::standard;

      prm.enter_subsection("Physical parameters");
      {
        Re  = prm.get_double("Re");
      }
      prm.leave_subsection();

      prm.enter_subsection("Time discretization parameters");
      {
        dt  = prm.get_double("dt");
        t_0 = prm.get_double("t_0");
        T   = prm.get_double("T");
      }
      prm.leave_subsection();

      prm.enter_subsection("Space discretization parameters");
      {
        n_global_refinements  = prm.get_integer("n_global_refinements");
        p_fe_degree           = prm.get_integer("p_fe_degree");
      }
      prm.leave_subsection();

      prm.enter_subsection("Parameters of the diffusion-step solver");
      {
        solver_max_iterations = prm.get_integer("solver_max_iterations");
        solver_tolerance      = prm.get_double("solver_tolerance");
        solver_krylov_size    = prm.get_integer("solver_krylov_size");
        solver_off_diagonals  = prm.get_integer("solver_off_diagonals");
        solver_diag_strength  = prm.get_double("solver_diag_strength");
        solver_update_preconditioner  = prm.get_integer(
          "solver_update_preconditioner");
      }
      prm.leave_subsection();

      flag_verbose_output = prm.get_bool("flag_verbose_output");
      output_interval     = prm.get_integer("output_interval");
    }
  } // namespace RunTimeParameters

  namespace EquationData
  {
    template <int dim>
    class VelocityIC : public Function<dim>
    {
    public:
      VelocityIC(const double time = 0)
        : Function<dim>(dim, time)
      {}
      virtual void vector_value(const Point<dim>  &p,
                                Vector<double>    &values) const override;
    };
    
    template <int dim>
    void VelocityIC<dim>::vector_value(const Point<dim>  &p,
                                      Vector<double>    &values) const
    {
        (void)p;
        values[0] = 0.0;
        values[1] = 0.0;
    }

    template <int dim>
    class VelocityInflowBC : public Function<dim>
    {
    public:
      VelocityInflowBC(const double time = 0)
        : Function<dim>(dim, time)
      {}
      virtual void vector_value(const Point<dim>  &p,
                                Vector<double>    &values) const override;
    };
    
    template <int dim>
    void VelocityInflowBC<dim>::vector_value(
                                        const Point<dim>  &p,
                                        Vector<double>    &values) const
    {
      const double Um = 1.5;
      const double H  = 4.1;

      values[0] = 4.0 * Um * p(1) * ( H - p(1) ) / ( H * H );
      values[1] = 0.0;
    }

    template <int dim>
    class PressureIC : public Function<dim>
    {
    public:
      PressureIC(const double time = 0)
        : Function<dim>(1, time)
      {}
      virtual double value(const Point<dim> &p,
                           const unsigned int component = 0) const override;
    };

    template<int dim>
    double PressureIC<dim>::value(const Point<dim> &p,
                                  const unsigned int component) const
    {
      (void)component;
      return 25.0 - p(0);
    }
  } // namespace EquationData

  namespace AdvectionTermAssembly
  {
    template <int dim>
    struct MappingData
    {
      FullMatrix<double>                   local_matrix;
      std::vector<types::global_dof_index> local_dof_indices;
      MappingData(const unsigned int dofs_per_cell)
        : local_matrix(dofs_per_cell, dofs_per_cell)
        , local_dof_indices(dofs_per_cell)
      {}
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
                    const UpdateFlags   flags)
        : fe_values(fe, quadrature_formula, flags),
          n_q_points(quadrature_formula.size()),
          dofs_per_cell(fe.dofs_per_cell),
          v_extrapolated_divergence(n_q_points),
          v_extrapolated_values(n_q_points),
          phi_v(dofs_per_cell),
          grad_phi_v(dofs_per_cell)
      {}

      LocalCellData(const LocalCellData &data)
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
                  const unsigned int p_dofs_per_cell)
        : v_dofs_per_cell(v_dofs_per_cell),
          p_dofs_per_cell(p_dofs_per_cell),
          local_matrix(v_dofs_per_cell, p_dofs_per_cell),
          v_local_dof_indices(v_dofs_per_cell),
          p_local_dof_indices(p_dofs_per_cell)
      {}
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
                    const UpdateFlags   p_flags)
        : v_fe_values(v_fe, quadrature_formula, v_flags),
          p_fe_values(p_fe, quadrature_formula, p_flags),
          n_q_points(quadrature_formula.size()),
          v_dofs_per_cell(v_fe.dofs_per_cell),
          p_dofs_per_cell(p_fe.dofs_per_cell),
          div_phi_v(v_dofs_per_cell),
          phi_p(p_dofs_per_cell)
      {}
      LocalCellData(const LocalCellData &data)
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
    };
  }// namespace PressureGradientTermAssembly

  template <int dim>
  class NavierStokesProjection
  {
  public:
    NavierStokesProjection(const RunTimeParameters::ParameterSet &data);
    void run(const bool flag_verbose_output = false, 
             const unsigned int n_plots = 10);

  protected:
    RunTimeParameters::ProjectionMethod       projection_method;
    const unsigned int                        p_fe_degree;
    const unsigned int                        v_fe_degree;
    const double                              dt;
    const double                              t_0;
    const double                              T;
    const double                              Re;

    EquationData::VelocityInflowBC<dim>       inflow_bc;
    EquationData::VelocityIC<dim>             v_initial_conditions;
    EquationData::PressureIC<dim>             p_initial_conditions;
    std::map<types::global_dof_index, double> boundary_values;
    std::vector<types::boundary_id>           boundary_ids;

    Triangulation<dim>                        triangulation;
    FESystem<dim>                             v_fe;
    FE_Q<dim>                                 p_fe;
    DoFHandler<dim>                           v_dof_handler;
    DoFHandler<dim>                           p_dof_handler;
    QGauss<dim>                               p_quadrature_formula;
    QGauss<dim>                               v_quadrature_formula;

    SparsityPattern                           v_sparsity_pattern;
    SparsityPattern                           p_sparsity_pattern;
    SparsityPattern                           mixed_sparsity_pattern;
    SparseMatrix<double>                      v_system_matrix;
    SparseMatrix<double>                      v_mass_matrix;
    SparseMatrix<double>                      v_laplace_matrix;
    SparseMatrix<double>                      v_mass_plus_laplace_matrix;
    SparseMatrix<double>                      v_advection_matrix;
    SparseMatrix<double>                      p_system_matrix;
    SparseMatrix<double>                      p_mass_matrix;
    SparseMatrix<double>                      p_laplace_matrix;
    SparseMatrix<double>                      p_gradient_matrix;

    Vector<double>                            v_n;
    Vector<double>                            v_n_m1;
    Vector<double>                            v_extrapolated;
    Vector<double>                            v_tmp;
    Vector<double>                            v_rot;
    Vector<double>                            v_rhs;
    Vector<double>                            p_n;
    Vector<double>                            p_n_m1;
    Vector<double>                            p_tmp;
    Vector<double>                            p_rhs;
    Vector<double>                            phi_n;
    Vector<double>                            phi_n_m1;

    SparseILU<double>                         v_preconditioner;
    SparseILU<double>                         p_preconditioner;
    SparseDirectUMFPACK                       p_update_preconditioner;
    SparseDirectUMFPACK                       v_rot_preconditioner;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void make_grid(const unsigned int n_global_refinements);
    void setup_dofs();
    void initialize();
    void diffusion_step_assembly();
    void diffusion_step_solve(const bool reinit_prec);
    void projection_step_assembly(const bool reinit_prec);
    void projection_step_solve(const bool reinit_prec);
    void correction_step(const bool reinit_prec);
    void output_results(const unsigned int step);

  private:
    using IteratorTuple =
      std::tuple<typename DoFHandler<dim>::active_cell_iterator,
                 typename DoFHandler<dim>::active_cell_iterator>;
    using IteratorPair = SynchronousIterators<IteratorTuple>;

    unsigned int                              solver_max_iterations;
    unsigned int                              solver_krylov_size;
    unsigned int                              solver_off_diagonals;
    unsigned int                              solver_update_preconditioner;
    double                                    solver_tolerance;
    double                                    solver_diag_strength;
    
    void setup_v_matrices();
    void setup_p_matrices();
    void setup_p_gradient_matrix();
    void assemble_v_matrices();
    void assemble_p_matrices();
    void assemble_p_gradient_matrix();
    void local_assemble_p_gradient_matrix(
      const IteratorPair                                    &SI,
      PressureGradientTermAssembly::LocalCellData<dim>      &scratch,
      PressureGradientTermAssembly::MappingData<dim>        &data);
    void mapping_p_gradient_matrix(
      const PressureGradientTermAssembly::MappingData<dim>  &data);

    void assemble_v_advection_matrix();
    void local_assemble_v_advection_matrix(
      const typename DoFHandler<dim>::active_cell_iterator  &cell,
      AdvectionTermAssembly::LocalCellData<dim>             &scratch,
      AdvectionTermAssembly::MappingData<dim>               &data);
    void mapping_v_advection_matrix(
      const AdvectionTermAssembly::MappingData<dim>         &data);
  };

  template <int dim>
  NavierStokesProjection<dim>::
  NavierStokesProjection(const RunTimeParameters::ParameterSet &data)
    : projection_method(data.projection_method),
      p_fe_degree(data.p_fe_degree),
      v_fe_degree(p_fe_degree + 1),
      dt(data.dt),
      t_0(data.t_0),
      T(data.T),
      Re(data.Re),
      inflow_bc(data.t_0),
      v_fe(FE_Q<dim>(v_fe_degree), dim),
      p_fe(p_fe_degree),
      v_dof_handler(triangulation),
      p_dof_handler(triangulation),
      p_quadrature_formula(p_fe_degree + 1),
      v_quadrature_formula(v_fe_degree + 1),
      solver_max_iterations(data.solver_max_iterations),
      solver_krylov_size(data.solver_krylov_size),
      solver_off_diagonals(data.solver_off_diagonals),
      solver_update_preconditioner(data.solver_update_preconditioner),
      solver_tolerance(data.solver_tolerance),
      solver_diag_strength(data.solver_diag_strength)
  {
    if (p_fe_degree < 1)
      std::cout
        << " WARNING: The chosen pair of finite element spaces is not stable."
        << std::endl
        << " The obtained results will be nonsense" << std::endl;

    AssertThrow(!((dt <= 0.) || (dt > .5 * T)), ExcInvalidTimeStep(dt, .5 * T));

    make_grid(data.n_global_refinements);
    setup_dofs();
    initialize();
  }

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

  template <int dim>
  void NavierStokesProjection<dim>::
  setup_dofs()
  {
    v_dof_handler.distribute_dofs(v_fe);
    DoFRenumbering::boost::Cuthill_McKee(v_dof_handler);
    p_dof_handler.distribute_dofs(p_fe);
    DoFRenumbering::boost::Cuthill_McKee(p_dof_handler);

    std::cout << "dim (X_h) = " << (v_dof_handler.n_dofs() )
          << std::endl
          << "dim (M_h) = " << p_dof_handler.n_dofs()
          << std::endl
          << "Re        = " << Re << std::endl
          << std::endl;
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  initialize()
  {
    setup_v_matrices();
    setup_p_matrices();
    setup_p_gradient_matrix();

    p_n.reinit(p_dof_handler.n_dofs());
    p_n_m1.reinit(p_dof_handler.n_dofs());
    p_rhs.reinit(p_dof_handler.n_dofs());
    p_tmp.reinit(p_dof_handler.n_dofs());
    phi_n.reinit(p_dof_handler.n_dofs());
    phi_n_m1.reinit(p_dof_handler.n_dofs());
    v_n.reinit(v_dof_handler.n_dofs());
    v_n_m1.reinit(v_dof_handler.n_dofs());
    v_extrapolated.reinit(v_dof_handler.n_dofs());
    v_rhs.reinit(v_dof_handler.n_dofs());
    v_tmp.reinit(v_dof_handler.n_dofs());

    assemble_v_matrices();
    assemble_p_matrices();
    assemble_p_gradient_matrix();

    v_mass_plus_laplace_matrix = 0.;
    v_mass_plus_laplace_matrix.add(1.0 / Re, v_laplace_matrix);
    v_mass_plus_laplace_matrix.add(1.5 / dt, v_mass_matrix);

    phi_n     = 0.;
    phi_n_m1  = 0.;

    p_initial_conditions.set_time(t_0);    
    VectorTools::interpolate(p_dof_handler, 
                             p_initial_conditions, 
                             p_n_m1);
    p_initial_conditions.advance_time(dt);
    VectorTools::interpolate(p_dof_handler, 
                             p_initial_conditions, 
                             p_n);

    v_initial_conditions.set_time(t_0);    
    VectorTools::interpolate(v_dof_handler,
                             v_initial_conditions,
                             v_n_m1);
    v_initial_conditions.advance_time(dt);
    VectorTools::interpolate(v_dof_handler,
                             v_initial_conditions,
                             v_n);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  setup_v_matrices()
  {
    {
      DynamicSparsityPattern dsp(v_dof_handler.n_dofs(),
                                 v_dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(v_dof_handler, dsp);
      v_sparsity_pattern.copy_from(dsp);
      std::ofstream out ("results_mod/v_sparsity_pattern.gpl");
      v_sparsity_pattern.print_gnuplot(out);
    }
    v_mass_plus_laplace_matrix.reinit(v_sparsity_pattern);
    v_system_matrix.reinit(v_sparsity_pattern);
    v_mass_matrix.reinit(v_sparsity_pattern);
    v_laplace_matrix.reinit(v_sparsity_pattern);
    v_advection_matrix.reinit(v_sparsity_pattern);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  setup_p_matrices()
  {
    {
      DynamicSparsityPattern dsp(p_dof_handler.n_dofs(),
                                 p_dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(p_dof_handler, dsp);
      p_sparsity_pattern.copy_from(dsp);
      std::ofstream out ("results_mod/p_sparsity_pattern.gpl");
      p_sparsity_pattern.print_gnuplot(out);
    }

    p_laplace_matrix.reinit(p_sparsity_pattern);
    p_system_matrix.reinit(p_sparsity_pattern);
    p_mass_matrix.reinit(p_sparsity_pattern);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  setup_p_gradient_matrix()
  {
    {
      DynamicSparsityPattern dsp(v_dof_handler.n_dofs(),
                                 p_dof_handler.n_dofs());
      DoFTools::make_sparsity_pattern(v_dof_handler,
                                      p_dof_handler,
                                      dsp);
      mixed_sparsity_pattern.copy_from(dsp);
      std::ofstream out("results_mod/mixed_sparsity_pattern.gpl");
      mixed_sparsity_pattern.print_gnuplot(out);
    }
    p_gradient_matrix.reinit(mixed_sparsity_pattern);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  assemble_v_matrices()
  {
    MatrixCreator::create_mass_matrix(v_dof_handler,
                                      v_quadrature_formula,
                                      v_mass_matrix);
    MatrixCreator::create_laplace_matrix(v_dof_handler,
                                         v_quadrature_formula,
                                         v_laplace_matrix);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  assemble_p_matrices()
  {
    MatrixCreator::create_laplace_matrix(p_dof_handler,
                                         p_quadrature_formula,
                                         p_laplace_matrix);
    MatrixCreator::create_mass_matrix(p_dof_handler,
                                      p_quadrature_formula,
                                      p_mass_matrix);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  assemble_p_gradient_matrix()
  {
    PressureGradientTermAssembly::MappingData<dim> per_task_data(
                                    v_fe.dofs_per_cell,
                                    p_fe.dofs_per_cell);
    PressureGradientTermAssembly::LocalCellData<dim> scratch_data(
                                    v_fe,
                                    p_fe,
                                    v_quadrature_formula,
                                    update_gradients | 
                                    update_JxW_values,
                                    update_values);

    WorkStream::run(
      IteratorPair(IteratorTuple(v_dof_handler.begin_active(),
                                 p_dof_handler.begin_active())),
      IteratorPair(IteratorTuple(v_dof_handler.end(),
                                 p_dof_handler.end())),
      *this,
      &NavierStokesProjection<dim>::local_assemble_p_gradient_matrix,
      &NavierStokesProjection<dim>::mapping_p_gradient_matrix,
      scratch_data,
      per_task_data);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  local_assemble_p_gradient_matrix(
    const IteratorPair & SI,
    PressureGradientTermAssembly::LocalCellData<dim>  &scratch,
    PressureGradientTermAssembly::MappingData<dim>    &data)
  {
    scratch.v_fe_values.reinit(std::get<0>(*SI));
    scratch.p_fe_values.reinit(std::get<1>(*SI));

    std::get<0>(*SI)->get_dof_indices(data.v_local_dof_indices);
    std::get<1>(*SI)->get_dof_indices(data.p_local_dof_indices);

    const FEValuesExtractors::Vector  velocity(0);

    data.local_matrix = 0.;
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    {
      for (unsigned int i = 0; i < data.v_dofs_per_cell; ++i)
        scratch.div_phi_v[i] = 
                        scratch.v_fe_values[velocity].divergence(i, q);
      for (unsigned int i = 0; i < data.p_dofs_per_cell; ++i)
        scratch.phi_p[i] = scratch.p_fe_values.shape_value(i,q);

      for (unsigned int i = 0; i < data.v_dofs_per_cell; ++i)
        for (unsigned int j = 0; j < data.p_dofs_per_cell; ++j)
          data.local_matrix(i, j) += -scratch.v_fe_values.JxW(q) *
                                      scratch.phi_p[j] *
                                      scratch.div_phi_v[i];
    }
  }


  template <int dim>
  void NavierStokesProjection<dim>::
  mapping_p_gradient_matrix(
    const PressureGradientTermAssembly::MappingData<dim> &data)
  {
    for (unsigned int i = 0; i < data.v_dofs_per_cell; ++i)
      for (unsigned int j = 0; j < data.p_dofs_per_cell; ++j)
        p_gradient_matrix.add(data.v_local_dof_indices[i],
                              data.p_local_dof_indices[j],
                              data.local_matrix(i, j));
  }

  template <int dim>
  void NavierStokesProjection<dim>::assemble_v_advection_matrix()
  {
    v_advection_matrix = 0.;
    AdvectionTermAssembly::MappingData<dim> data(
                                v_fe.dofs_per_cell);
    AdvectionTermAssembly::LocalCellData<dim> scratch(
                                v_fe,
                                v_quadrature_formula,
                                update_values | 
                                update_JxW_values |
                                update_gradients);
    WorkStream::run(
      v_dof_handler.begin_active(),
      v_dof_handler.end(),
      *this,
      &NavierStokesProjection<dim>::local_assemble_v_advection_matrix,
      &NavierStokesProjection<dim>::mapping_v_advection_matrix,
      scratch,
      data);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  local_assemble_v_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    AdvectionTermAssembly::LocalCellData<dim>             &scratch,
    AdvectionTermAssembly::MappingData<dim>               &data)
  {
    scratch.fe_values.reinit(cell);
    cell->get_dof_indices(data.local_dof_indices);
    const FEValuesExtractors::Vector      velocity(0);

    scratch.fe_values[velocity].get_function_values(
                                        v_extrapolated, 
                                        scratch.v_extrapolated_values);
    scratch.fe_values[velocity].get_function_divergences(
                                    v_extrapolated, 
                                    scratch.v_extrapolated_divergence);

    data.local_matrix = 0.;
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
    {
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      {
        scratch.phi_v[i] = scratch.fe_values[velocity].value(i,q);
        scratch.grad_phi_v[i] = scratch.fe_values[velocity].gradient(i,q);
      }
      for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
          data.local_matrix(i, j) += ( 
                              scratch.phi_v[i] *
                              scratch.grad_phi_v[j] *  
                              scratch.v_extrapolated_values[q]            
                              +                                    
                              0.5 *                                
                              scratch.v_extrapolated_divergence[q] *            
                              scratch.phi_v[i] * 
                              scratch.phi_v[j])  
                              * scratch.fe_values.JxW(q);
    }
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  mapping_v_advection_matrix(
    const AdvectionTermAssembly::MappingData<dim> &data)
  {
    for (unsigned int i = 0; i < v_fe.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < v_fe.dofs_per_cell; ++j)
        v_advection_matrix.add(data.local_dof_indices[i],
                          data.local_dof_indices[j],
                          data.local_matrix(i, j));
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  diffusion_step_assembly()
  {
    /*Extrapolate velocity by a Taylor expansion
      v^{\textrm{k}+1} \approx 2 * v^\textrm{k} - v^{\textrm{k}-1 */

    v_extrapolated.equ(2., v_n);
    v_extrapolated -= v_n_m1;

    /*Define auxiliary pressure
      p^{\#} = p^\textrm{k} + 4/3 * \phi^\textrm{k} 
                - 1/3 * \phi^{\textrm{k}-1} 
      Note: The signs are inverted since p_gradient_matrix is
      defined as negative */

    p_tmp.equ(-1., p_n);
    p_tmp.add(-4. / 3., phi_n, 1. / 3., phi_n_m1);

    /* System matrix setup */
    assemble_v_advection_matrix();
    v_system_matrix.copy_from(v_mass_plus_laplace_matrix);
    v_system_matrix.add(1., v_advection_matrix);

    /* Right hand side setup */
    v_rhs = 0.;
    v_tmp.equ(2. / dt, v_n);
    v_tmp.add(-.5 / dt, v_n_m1);
    v_mass_matrix.vmult_add(v_rhs, v_tmp);
    p_gradient_matrix.vmult_add(v_rhs, p_tmp);

    /* Update for the next time step */
    v_n_m1 = v_n;

    boundary_values.clear();
    for (const auto &boundary_id : boundary_ids)
      {
        switch (boundary_id)
          {
            case 1:
              VectorTools::interpolate_boundary_values(
                v_dof_handler,
                boundary_id,
                Functions::ZeroFunction<dim>(dim),
                boundary_values);
              break;
            case 2:
              VectorTools::interpolate_boundary_values(v_dof_handler,
                                                        boundary_id,
                                                        inflow_bc,
                                                        boundary_values);
              break;
            case 3:
            {
              ComponentMask   component_mask(dim, true);
              component_mask.set(0, false);
              VectorTools::interpolate_boundary_values(
                v_dof_handler,
                boundary_id,
                Functions::ZeroFunction<dim>(dim),
                boundary_values,
                component_mask);
              break;
            }
            case 4:
              VectorTools::interpolate_boundary_values(
                v_dof_handler,
                boundary_id,
                Functions::ZeroFunction<dim>(dim),
                boundary_values);
              break;
            default:
              Assert(false, ExcNotImplemented());
          }
      }
    MatrixTools::apply_boundary_values(boundary_values,
                                        v_system_matrix,
                                        v_n,
                                        v_rhs);
  }

  template <int dim>
  void
  NavierStokesProjection<dim>::
  diffusion_step_solve(const bool reinit_prec)
  {
    if (reinit_prec)
      v_preconditioner.initialize(v_system_matrix,
                                      SparseILU<double>::AdditionalData(
                                        solver_diag_strength, 
                                        solver_off_diagonals));

    SolverControl solver_control(solver_max_iterations, 
                                 solver_tolerance * v_rhs.l2_norm());
    SolverGMRES<> gmres(solver_control,
                        SolverGMRES<>::AdditionalData(solver_krylov_size));
    gmres.solve(v_system_matrix, 
                v_n, 
                v_rhs, 
                v_preconditioner);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  projection_step_assembly(const bool reinit_prec)
  {
    /* System matrix setup */
    p_system_matrix.copy_from(p_laplace_matrix);

    /* Right hand side setup */
    p_rhs = 0.;
    p_gradient_matrix.Tvmult_add(p_rhs, v_n);

    /* Update for the next time step */
    phi_n_m1 = phi_n;

    static std::map<types::global_dof_index, double> bval;
    if (reinit_prec)
      VectorTools::interpolate_boundary_values(p_dof_handler,
                                               3,
                                               Functions::ZeroFunction<dim>(),
                                               bval);

    MatrixTools::apply_boundary_values(bval, 
                                       p_system_matrix, 
                                       phi_n, 
                                       p_rhs);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  projection_step_solve(const bool reinit_prec)
  {
    if (reinit_prec)
      p_preconditioner.initialize(p_system_matrix,
                                   SparseILU<double>::AdditionalData(
                                     solver_diag_strength, 
                                     solver_off_diagonals));

    SolverControl solvercontrol(solver_max_iterations, 
                                solver_tolerance * p_rhs.l2_norm());
    SolverCG<>    cg(solvercontrol);
    cg.solve(p_system_matrix, 
             phi_n, 
             p_rhs, 
             p_preconditioner);

    phi_n *= 1.5 / dt;
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  correction_step(const bool reinit_prec)
  {
    /* Update for the next time step */
    p_n_m1 = p_n;
    switch (projection_method)
      {
        case RunTimeParameters::ProjectionMethod::standard:
          p_n += phi_n;
          break;
        case RunTimeParameters::ProjectionMethod::rotational:
          if (reinit_prec)
            p_update_preconditioner.initialize(p_mass_matrix);
          p_n = p_rhs;
          p_update_preconditioner.solve(p_n);
          p_n.sadd(1. / Re, 1., p_n_m1);
          p_n += phi_n;
          break;
        default:
          Assert(false, ExcNotImplemented());
      };
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  output_results(const unsigned int step)
  {
    std::vector<std::string> names(dim, "velocity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim>        data_out;

    data_out.add_data_vector(v_dof_handler, 
                             v_n, 
                             names, 
                             component_interpretation);
    data_out.add_data_vector(p_dof_handler, 
                             p_n, 
                             "Pressure");
    data_out.build_patches(p_fe_degree+1);

    std::ofstream       output_file("results_mod/solution-" 
                                    + Utilities::int_to_string(step, 5)
                                    + ".vtk");

    data_out.write_vtk(output_file);
  }

  template <int dim>
  void NavierStokesProjection<dim>::
  run(const bool  flag_verbose_output,
      const unsigned int output_interval)
  {
    ConditionalOStream verbose_cout(std::cout, flag_verbose_output);

    const auto n_steps = static_cast<unsigned int>((T - t_0) / dt);

    inflow_bc.set_time(2. * dt);
    output_results(1);

    for (unsigned int n = 2; n <= n_steps; ++n)
      {
        if (n % output_interval == 0)
          {
            verbose_cout << "Plotting Solution" << std::endl;
            output_results(n);
          }
        std::cout << "Step = " << n 
                  << " Time = " << (n * dt) << std::endl;

        verbose_cout << "  Diffusion Step" << std::endl;
        if (n % solver_update_preconditioner == 0)
          verbose_cout << "    With reinitialization of the preconditioner"
                       << std::endl;
        diffusion_step_assembly();
        diffusion_step_solve((n % solver_update_preconditioner == 0) || (n == 2));

        verbose_cout << "  Projection Step" << std::endl;
        projection_step_assembly((n == 2));
        projection_step_solve((n==2));
        
        verbose_cout << "  Updating the Pressure" << std::endl;
        correction_step((n == 2));

        inflow_bc.advance_time(dt);
      }
    output_results(n_steps);
  }

} // namespace Step35

int main()
{
  try
    {
      using namespace dealii;
      using namespace Step35;

      RunTimeParameters::ParameterSet parameter_set;
      parameter_set.read_data_from_file("parameter_file.prm");

      deallog.depth_console(parameter_set.flag_verbose_output ? 2 : 0);

      NavierStokesProjection<2> simulation(parameter_set);
      simulation.run(parameter_set.flag_verbose_output, 
                     parameter_set.output_interval);
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
