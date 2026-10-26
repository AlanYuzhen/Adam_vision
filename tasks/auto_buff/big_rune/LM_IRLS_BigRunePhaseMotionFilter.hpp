#pragma once

// LM+IRLS 大符相位运动拟合器（迁移自 RP-26Rune LM_IRLS_BigRunePhaseMotionFilter.hpp）
// 外层 LM 迭代 ω，内层 IRLS 迭代 A,B,b,C；参数经构造注入。

#include "big_rune/BigRunePhaseMotionFilter.hpp"

#include <limits>

namespace big_rune
{

class LM_IRLS_BigRunePhaseMotionFilter : public BigRunePhaseMotionFilter
{
private:
  // 内层优化结构体
  struct FitResultInside
  {
    Eigen::Vector4d theta = Eigen::Vector4d::Zero();          // A,B,b,C
    double cost = std::numeric_limits<double>::infinity();    // 默认代价无限大
    bool is_valid = false;
  };

  FitResultInside m_fit_result_inside;                       // 内层优化结构体
  void fit_motion_model() override;  // 拟合运动方程(LM迭代ω IRLS迭代,A,B,b,C)
  void prepare_date();               // 准备拟合所需要的数据(计算去中心化时间戳和时间权重)
  void fit_linear(const double & omega);  // 内层线性项迭代(IRLS)
  void complete_motion(const Eigen::Vector4d & theta, const double & omega);  // 用拟合结果完善运动方程

  Eigen::VectorXd m_t;                // 时间数据
  Eigen::VectorXd m_y;                // 相位数据
  Eigen::VectorXd m_time_weight;      // 时间权重(越新的数据权重越大)
  Eigen::VectorXd m_residual_weight;  // 残差权重(残差越小数据权重越大)
  Eigen::VectorXd m_total_weight;     // 和权重(时间权重和残差权重的内积)
  Eigen::Matrix4d m_ridge = 1e-6 * Eigen::Matrix4d::Identity();  // 防止海森矩阵奇异

public:
  explicit LM_IRLS_BigRunePhaseMotionFilter(const MotionFilterParams & cfg)
  : BigRunePhaseMotionFilter(cfg)
  {
  }
};

}  // namespace big_rune
