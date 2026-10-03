// Spike: the tiny-solver pose-graph objective (same residuals as BetweenFactorSE2/SE3
// and PriorFactor) solved with ceres. Reads a problem dumped by robust_pgo, writes
// the optimized vertices. usage: pgo <problem> <none|huber|cauchy> <scale> <default|tight> <out>
#include <ceres/ceres.h>
#include <ceres/manifold.h>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

template <typename T> void QMul(const T* a, const T* b, T* c) {  // xyzw
  c[3] = a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2];
  c[0] = a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1];
  c[1] = a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0];
  c[2] = a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3];
}
template <typename T> void QRot(const T* q, const T* v, T* out) {  // q (v,0) conj(q)
  T qv[4] = {v[0], v[1], v[2], T(0)}, qc[4] = {-q[0], -q[1], -q[2], q[3]}, t[4], r[4];
  QMul(q, qv, t);
  QMul(t, qc, r);
  out[0] = r[0]; out[1] = r[1]; out[2] = r[2];
}
template <typename T> void QLog(const T* q, T* out) {  // as tiny-solver SO3::log
  T n2 = q[0] * q[0] + q[1] * q[1] + q[2] * q[2];
  T w = q[3], f;
  if (n2 <= T(1e-12)) {
    f = T(2.0) / w - T(2.0 / 3.0) * n2 / (w * w * w);
  } else {
    T n = sqrt(n2);
    T sign = w <= T(0) ? T(-1) : T(1);
    f = T(2.0) * sign * atan2(n, sign * w) / n;
  }
  out[0] = f * q[0]; out[1] = f * q[1]; out[2] = f * q[2];
}

struct Se3Between {
  double m[7];  // dqx dqy dqz dqw dtx dty dtz
  template <typename T> bool operator()(const T* p0, const T* p1, T* r) const {
    T dq[4] = {T(m[0]), T(m[1]), T(m[2]), T(m[3])}, dt[3] = {T(m[4]), T(m[5]), T(m[6])};
    T qa[4], ta[3];
    QMul(p0, dq, qa);
    QRot(p0, dt, ta);
    for (int i = 0; i < 3; ++i) ta[i] += p0[4 + i];
    T q1c[4] = {-p1[0], -p1[1], -p1[2], p1[3]}, qd[4], diff[3], td[3];
    QMul(q1c, qa, qd);
    for (int i = 0; i < 3; ++i) diff[i] = ta[i] - p1[4 + i];
    QRot(q1c, diff, td);
    QLog(qd, r);
    r[3] = td[0]; r[4] = td[1]; r[5] = td[2];
    return true;
  }
};
struct Se2Between {
  double dx, dy, dth;  // params: theta x y
  template <typename T> bool operator()(const T* p0, const T* p1, T* r) const {
    T c0 = cos(p0[0]), s0 = sin(p0[0]), c1 = cos(p1[0]), s1 = sin(p1[0]);
    T ax = p0[1] + c0 * dx - s0 * dy - p1[1], ay = p0[2] + s0 * dx + c0 * dy - p1[2];
    r[0] = c1 * ax + s1 * ay;
    r[1] = -s1 * ax + c1 * ay;
    T a = p0[0] + dth - p1[0];
    r[2] = atan2(sin(a), cos(a));
    return true;
  }
};
struct Prior {
  std::vector<double> v;
  template <typename T> bool operator()(T const* const* p, T* r) const {
    for (size_t i = 0; i < v.size(); ++i) r[i] = p[0][i] - v[i];
    return true;
  }
};

int main(int argc, char** argv) {
  std::ifstream in(argv[1]);
  std::string loss = argv[2], tol = argv[4];
  double scale = std::stod(argv[3]);
  std::map<int, std::vector<double>> x;
  std::vector<int> order;
  ceres::Problem problem;
  std::string line;
  auto make_loss = [&]() -> ceres::LossFunction* {
    if (loss == "huber") return new ceres::HuberLoss(scale);
    if (loss == "cauchy") return new ceres::CauchyLoss(scale);
    return nullptr;
  };
  bool is3d = false;
  std::vector<std::string> lines;
  while (std::getline(in, line)) lines.push_back(line);
  for (auto& l : lines) {
    std::istringstream s(l);
    std::string tag; s >> tag;
    if (tag == "V") { int id; s >> id; double v; while (s >> v) x[id].push_back(v); order.push_back(id); }
  }
  for (auto& l : lines) {
    std::istringstream s(l);
    std::string tag; s >> tag;
    if (tag == "E2") {
      int a, b; auto* f = new Se2Between; s >> a >> b >> f->dx >> f->dy >> f->dth;
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Se2Between, 3, 3, 3>(f), make_loss(), x[a].data(), x[b].data());
    } else if (tag == "E3") {
      is3d = true;
      int a, b; auto* f = new Se3Between; double d[7]; s >> a >> b;
      for (double& v : d) s >> v;  // dtx dty dtz dqx dqy dqz dqw
      f->m[0] = d[3]; f->m[1] = d[4]; f->m[2] = d[5]; f->m[3] = d[6]; f->m[4] = d[0]; f->m[5] = d[1]; f->m[6] = d[2];
      problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Se3Between, 6, 7, 7>(f), make_loss(), x[a].data(), x[b].data());
    } else if (tag == "P") {
      int a; s >> a; auto* f = new Prior; double v; while (s >> v) f->v.push_back(v);
      auto* c = new ceres::DynamicAutoDiffCostFunction<Prior>(f);
      c->AddParameterBlock(f->v.size()); c->SetNumResiduals(f->v.size());
      problem.AddResidualBlock(c, make_loss(), x[a].data());
    }
  }
  if (is3d)
    for (auto& [id, v] : x)
      problem.SetManifold(v.data(), new ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>());
  ceres::Solver::Options o;
  o.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  o.max_num_iterations = 100;
  if (tol == "tight") { o.function_tolerance = o.gradient_tolerance = o.parameter_tolerance = 1e-16; o.max_num_iterations = 500; }
  ceres::Solver::Summary sum;
  ceres::Solve(o, &problem, &sum);
  std::ofstream out(argv[5]);
  out.precision(17);
  for (auto& [id, v] : x) { out << id; for (double d : v) out << " " << d; out << "\n"; }
  std::printf("ceres %s iterations %d time %.3f initial_sum_rho %.9e final_sum_rho %.12e %s\n", ceres::TerminationTypeToString(sum.termination_type),
              sum.num_successful_steps + sum.num_unsuccessful_steps, sum.total_time_in_seconds, 2 * sum.initial_cost, 2 * sum.final_cost, sum.message.c_str());
}
