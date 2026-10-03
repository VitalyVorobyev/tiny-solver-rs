// Spike: solve the dumped robust-fit instances with Ceres (LM, tight tolerances).
#include <ceres/ceres.h>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct Fit {
  Fit(int m, double x, double y) : m(m), x(x), y(y) {}
  template <typename T> bool operator()(const T* p, T* r) const {
    T f = m == 0 ? p[0] * x + p[1] : p[0] * exp(-(p[1] * x)) + p[2];
    r[0] = f - y;
    return true;
  }
  int m; double x, y;
};

int main(int argc, char** argv) {
  bool tight = argc < 2 || std::string(argv[1]) == "tight";
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    int seed, model, loss, ninit, npts; double a;
    in >> seed >> model >> loss >> a >> ninit;
    std::vector<double> p(ninit);
    for (auto& v : p) in >> v;
    in >> npts;
    ceres::Problem problem;
    std::vector<std::pair<double,double>> pts(npts);
    for (auto& xy : pts) {
      in >> xy.first >> xy.second;
      ceres::LossFunction* l = loss == 0 ? (ceres::LossFunction*)new ceres::HuberLoss(a)
                             : loss == 1 ? (ceres::LossFunction*)new ceres::CauchyLoss(a)
                                         : new ceres::ArctanLoss(a);
      ceres::CostFunction* c = model == 0
          ? (ceres::CostFunction*)new ceres::AutoDiffCostFunction<Fit, 1, 2>(new Fit(model, xy.first, xy.second))
          : new ceres::AutoDiffCostFunction<Fit, 1, 3>(new Fit(model, xy.first, xy.second));
      problem.AddResidualBlock(c, l, p.data());
    }
    ceres::Solver::Options o;
    o.linear_solver_type = ceres::DENSE_QR;
    if (tight) { o.function_tolerance = o.gradient_tolerance = o.parameter_tolerance = 1e-16; o.max_num_iterations = 500; }
    ceres::Solver::Summary s;
    ceres::Solve(o, &problem, &s);
    std::printf("%d %d %d ceres %.12e", seed, model, loss, 2.0 * s.final_cost);
    for (double v : p) std::printf(" %.10f", v);
    std::printf(" %s\n", ceres::TerminationTypeToString(s.termination_type));
  }
}
