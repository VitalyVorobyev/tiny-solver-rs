// Spike: the same robust line fit in Ceres. Usage: robust_fit <csv> <huber|cauchy> <default|tight> [a0 b0]
#include <ceres/ceres.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

struct Line {
  Line(double x, double y) : x(x), y(y) {}
  template <typename T> bool operator()(const T* ab, T* r) const {
    r[0] = ab[0] * x + ab[1] - y;
    return true;
  }
  double x, y;
};

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::string kind = argv[2], tol = argv[3];
  double ab[2] = {argc > 5 ? std::stod(argv[4]) : 0.0, argc > 5 ? std::stod(argv[5]) : 0.0};
  const double scale = 0.5;
  ceres::Problem problem;
  std::vector<std::pair<double, double>> pts;
  std::string line;
  while (std::getline(in, line)) {
    auto c = line.find(',');
    pts.emplace_back(std::stod(line.substr(0, c)), std::stod(line.substr(c + 1)));
  }
  for (auto& p : pts) {
    ceres::LossFunction* loss = kind == "huber" ? (ceres::LossFunction*)new ceres::HuberLoss(scale)
                                                : new ceres::CauchyLoss(scale);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Line, 1, 2>(new Line(p.first, p.second)),
                             loss, ab);
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_QR;
  if (tol == "tight") {
    options.function_tolerance = options.gradient_tolerance = options.parameter_tolerance = 1e-16;
    options.max_num_iterations = 500;
  }
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::printf("%s ceres %s: a %.12f b %.12f sum_rho %.12f iters %d %s\n", kind.c_str(), tol.c_str(),
              ab[0], ab[1], 2.0 * summary.final_cost,
              summary.num_successful_steps + summary.num_unsuccessful_steps,
              ceres::TerminationTypeToString(summary.termination_type));
}
