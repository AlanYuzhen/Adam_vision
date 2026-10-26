#pragma once

// 大符相位运动观测器基类（迁移自 RP-26Rune BigRunePhaseMotionFilter.hpp）
// 改造点：foxglove::Time -> big_rune::Time；J_POWER_RUNE 全局配置 -> 构造注入 MotionFilterParams；
//         移除 Viz/日志依赖，包裹 namespace big_rune。算法逻辑与 RP 原版逐行对应。

#include "big_rune/types.hpp"

#include <deque>
#include <vector>

namespace big_rune
{

class BigRunePhaseMotionFilter
{
  // 1.获取相位数据
  // 2.判断运动方程是否已经建立
  //     没有建立：
  //         将数据加入队列
  //         判断队列数目是否足以拟合方程
  //         不允许则继续
  //         允许则对数据进行分离，然后进行数据连续化
  //         拟合方程
  //     已经建立：
  //         判断是单个目标还是两个目标
  //             单个目标：
  //                 对数据进行连续化处理，迭代方程
  //             两个目标：
  //                 进行预测后分别与两个目标的相位进行对比
  //                     如果有在预测连续范围内的目标，则选择该目标，然后进行数据连续化
  //                     如果没有则选择跳变较小的目标进行数据连续化,然后对数据进行连续化
  //                 迭代运动方程
public:
  explicit BigRunePhaseMotionFilter(const MotionFilterParams & cfg);
  virtual ~BigRunePhaseMotionFilter() = default;

  bool calculate_motion(std::vector<CandidateTarget> && candidate_targets);  // 拟合运动方程
  // 追踪队列是否非空。建模前（数据量不足 min_data_size）队列为空，
  // 此时 get_target() 不可用——调用方必须先检查本接口。
  bool has_tracked_target() const { return !m_tracked_target_deque.empty(); }
  virtual const CandidateTarget get_target() const;                          // 获取目标
  virtual const MotionModelParams get_motion() const;                        // 获取运动方程
  void set_rotation_direction(RotationDirection rotation_direction);         // 设置旋转方向

protected:
  MotionFilterParams cfg_;  // 参数（RP 原版直读 J_POWER_RUNE，此处构造注入）

  std::deque<std::vector<CandidateTarget>> m_ori_candidate_targets_deque;  // 候选目标队列(原始未分离的数据)
  void update_ori_candidate_targets(std::vector<CandidateTarget> && ori_candidate_targets);  // 更新候选目标队列
  bool try_build_motion_model();  // 尝试构建运动方程
  bool try_update_motion_model();  // 尝试更新运动方程
  bool confirm_rotation();         // 确定旋转方向
  void init_tracked_target_deque();  // 初始化追踪目标队列(即进行数据连续化)
  int choose_tracked_target(const std::vector<CandidateTarget> & ori_candidate_targets);  // 选择需要追踪的目标
  bool is_phase_jump(const CandidateTarget & ori_candidate_target);  // 相位是否出现了跳变
  double guess_phase(Time future_time);  // 在没有运动方程的时候猜测在future_time时的相位
  void update_tracked_target_deque();  // 更新追踪目标队列
  void resolve_track_after_phase_jump(const CandidateTarget & ori_candidate_target);  // 目标切换的时候重新计算连续相位
  double accumulate_phase(const double & old_phase, const double & new_phase);  // 单步累加，用于连续化角度。输入旧的角度，新的角度。返回旧的角度的累加
  virtual void fit_motion_model() = 0;  // 拟合运动模型

  MotionModelParams m_big_rune_motion_model;  // 大符的运动方程
  bool m_is_motion_model_vaild = false;       // 运动方程是否有效
  RotationDirection m_rotation_direction = RotationDirection::Unknown;  // 旋转方向
  std::deque<TrackedTarget> m_tracked_target_deque;  // 经过匹配算法得到的追踪的目标(已经经过匹配和连续化后的数据)

  struct Jump
  {
    bool is_jump;
    Time jump_time;
  };
  std::deque<Jump> m_jump_deque;
  bool need_reset_motion_for_abnormal_jump();

  void filter_with_motion();  // 使用运动方程进行滤波
};

}  // namespace big_rune
