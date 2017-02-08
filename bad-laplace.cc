#include <deal.II/base/function.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>

#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_parallel_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>
#include <string>
#include <memory>
#include <vector>


using namespace dealii;

constexpr int fe_degree {5};
constexpr types::manifold_id circular_manifold_id {0};
constexpr types::manifold_id straight_manifold_id {dealii::numbers::flat_manifold_id};

constexpr types::boundary_id circular_boundary_id {2};

// laziness
constexpr double pi = M_PI;
constexpr double signal_nan {std::numeric_limits<double>::signaling_NaN()};

template <int dim>
class HardManufacturedSolution : public Function<dim>
{
public:
  virtual double value(const Point<dim> &point,
                       const unsigned int) const override
  {
    static_assert(dim == 2, "only available in 2D");

    return point[0] * point[1];
  }
};



template <int dim>
class HardManufacturedForcing : public Function<dim>
{
public:
  virtual double value(const Point<dim> &point,
                       const unsigned int) const override
  {
    static_assert(dim == 2, "not implemented for dim != 2");
    (void)point;

    return 0.0;
  }
};



template <int dim>
class BadLaplace
{
public:
  BadLaplace(const unsigned int n_global_refines);

  double run();
  void save_grid();

protected:
  const unsigned int n_global_refines;

  std::unique_ptr<Function<dim>> manufactured_solution;
  std::unique_ptr<Function<dim>> manufactured_forcing;

  std::unique_ptr<Manifold<dim>> boundary_manifold;
  Triangulation<dim> triangulation;
  FE_Q<dim> finite_element;
  DoFHandler<dim> dof_handler;
  QGauss<dim> cell_quadrature;
  MappingQ<dim> mapping;

  ConstraintMatrix constraints;
  SparsityPattern sparsity_pattern;
  PETScWrappers::MPI::SparseMatrix system_matrix;
  PETScWrappers::MPI::Vector system_rhs;
  PETScWrappers::MPI::Vector solution;

  void setup_matrices();
  double solve();
};


template <int dim>
BadLaplace<dim>::BadLaplace(const unsigned int n_global_refines) :
  n_global_refines {n_global_refines},
  manufactured_solution {new HardManufacturedSolution<dim>()},
  manufactured_forcing {new HardManufacturedForcing<dim>()},
  boundary_manifold {new SphericalManifold<dim>()},
  finite_element(fe_degree),
  dof_handler(triangulation),
  cell_quadrature(fe_degree + 1),
  mapping(fe_degree)
{
  GridGenerator::hyper_shell(triangulation, Point<dim>(), 1.0, 2.0);
  triangulation.set_all_manifold_ids(circular_manifold_id);
  triangulation.set_manifold(circular_manifold_id, *boundary_manifold);
  triangulation.refine_global(n_global_refines);

  triangulation.set_all_manifold_ids(straight_manifold_id);
  triangulation.set_all_manifold_ids_on_boundary(circular_manifold_id);

  for (auto cell : triangulation.active_cell_iterators())
    {
      for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell;
           ++face_n)
        {
          if (cell->face(face_n)->at_boundary())
            {
              cell->face(face_n)->set_boundary_id(circular_boundary_id);
            }
        }
    }

  dof_handler.distribute_dofs(finite_element);
  VectorTools::interpolate_boundary_values(mapping,
                                           dof_handler,
                                           circular_boundary_id,
                                           *manufactured_solution,
                                           constraints);
  constraints.close();
}



template <int dim>
void BadLaplace<dim>::setup_matrices()
{
  {
    DynamicSparsityPattern dynamic_sparsity_pattern(dof_handler.n_dofs(),
                                                    dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dynamic_sparsity_pattern,
                                    constraints, false);
    sparsity_pattern.copy_from(dynamic_sparsity_pattern);
  }
  IndexSet all_dofs(dof_handler.n_dofs());
  all_dofs.add_range(0, dof_handler.n_dofs());
  all_dofs.compress();

  system_matrix.reinit(all_dofs, all_dofs, sparsity_pattern, MPI_COMM_WORLD);
  system_rhs.reinit(all_dofs, MPI_COMM_WORLD);
  solution.reinit(all_dofs, MPI_COMM_WORLD);

  const UpdateFlags flags {update_values | update_gradients | update_JxW_values
      | update_quadrature_points};
  FEValues<dim> fe_values(mapping, finite_element, cell_quadrature, flags);
  const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
  FullMatrix<double> cell_system(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  for (const auto cell : dof_handler.active_cell_iterators())
    {
      cell->get_dof_indices(local_dof_indices);
      cell_system = 0.0;
      cell_rhs = 0.0;
      fe_values.reinit(cell);

      for (unsigned int q_point_n = 0;
           q_point_n < fe_values.n_quadrature_points; ++q_point_n)
        {
          const double point_forcing
            {manufactured_forcing->value(fe_values.quadrature_point(q_point_n))};

          for (unsigned int test_n = 0; test_n < dofs_per_cell; ++test_n)
            {
              for (unsigned int trial_n = 0; trial_n < dofs_per_cell; ++trial_n)
                {
                  cell_system(test_n, trial_n) += fe_values.JxW(q_point_n)*
                    (fe_values.shape_grad(test_n, q_point_n)
                     *fe_values.shape_grad(trial_n, q_point_n));
                  // The projection also fails to converge at the correct
                  // rate. To see this, set the manufactured solution equal to
                  // the exact solution.
                  // cell_system(test_n, trial_n) += fe_values.JxW(q_point_n)*
                  //   (fe_values.shape_value(test_n, q_point_n)
                  //    *fe_values.shape_value(trial_n, q_point_n));
                }

              cell_rhs[test_n] += fe_values.JxW(q_point_n)
                *fe_values.shape_value(test_n, q_point_n)
                *point_forcing;
            }
        }
      constraints.distribute_local_to_global(cell_system, cell_rhs,
                                             local_dof_indices,
                                             system_matrix, system_rhs);
    }
  system_matrix.compress(VectorOperation::add);
  system_rhs.compress(VectorOperation::add);
}



template <int dim>
double BadLaplace<dim>::solve()
{
  SolverControl solver_control(std::max(types::global_dof_index(100),
                                        system_rhs.size()),
                               1e-14*system_rhs.l2_norm(),
                               /*log_history =*/ false,
                               /*log_result =*/ false);

  PETScWrappers::SolverCG solver(solver_control, MPI_COMM_WORLD);
  PETScWrappers::PreconditionBoomerAMG::AdditionalData preconditioner_options;
  preconditioner_options.symmetric_operator = true;
  PETScWrappers::PreconditionBoomerAMG preconditioner(system_matrix,
                                                      preconditioner_options);
  solution.reinit(system_rhs);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);
  constraints.distribute(solution);

  Vector<double> cell_l2_error(triangulation.n_cells());
  {
    QIterated<dim> cell_error_quadrature(QGauss<1>(fe_degree + 1), 2);
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      *manufactured_solution,
                                      cell_l2_error,
                                      cell_error_quadrature,
                                      VectorTools::NormType::L2_norm);
  }

  std::sort(cell_l2_error.begin(), cell_l2_error.end());
  double l2_error = 0.0;
  for (const double value : cell_l2_error)
    {
      l2_error += value*value;
    }

  return std::sqrt(l2_error);
}



template <int dim>
double BadLaplace<dim>::run()
{
  setup_matrices();
  const double l2_error = solve();
  save_grid();

  return l2_error;
}



template <int dim>
void BadLaplace<dim>::save_grid()
{
  DataOut<dim, DoFHandler<dim>> data_out;
  data_out.attach_dof_handler(dof_handler);

  // also save the error
  QIterated<dim> cell_error_quadrature(QGauss<1>(fe_degree + 1), 2);

  Vector<double> cell_linfty_error(triangulation.n_cells());
  {
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      solution,
                                      *manufactured_solution,
                                      cell_linfty_error,
                                      cell_error_quadrature,
                                      VectorTools::NormType::Linfty_norm);

    for (double &entry : cell_linfty_error)
      {
        entry = std::log10(entry + 1.0e-16);
      }
    data_out.add_data_vector(cell_linfty_error, "log10_of_Linf_error");
  }

  data_out.build_patches();
  DataOutBase::VtkFlags vtk_flags;
#if DEAL_II_VERSION_GTE(8, 4, 0)
  vtk_flags.compression_level = DataOutBase::VtkFlags::ZlibCompressionLevel::best_speed;
#endif
  data_out.set_flags(vtk_flags);
  std::ofstream output("grid-" + Utilities::int_to_string(dof_handler.n_dofs()) + ".vtu");
  data_out.write_vtu(output);
}



int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  constexpr int dim {2};

  std::vector<double> errors;
  for (unsigned int n_global_refines = 0; n_global_refines < 8;
       ++n_global_refines)
    {
      BadLaplace<dim> laplace_solver(n_global_refines);
      errors.push_back(laplace_solver.run());

      if (errors.size() > 1)
        {
          const double &previous = *(errors.end() - 2);
          const double &current = *(errors.end() - 1);

          // make mesh widths differ by a factor of two: the exact value
          // doesn't really matter since we always refine uniformly
          const double denominator {std::log10(std::pow(2.0, -double(errors.size())))
              - std::log10(std::pow(2.0, -double(errors.size() + 1)))};
          std::cout << "L2 error: " << current << '\n';
          std::cout << "L2 slope: "
                    << (std::log10(previous) - std::log10(current))
                       /denominator
                    << '\n';
        }
    }
}
