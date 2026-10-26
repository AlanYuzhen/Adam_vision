#include <fmt/format.h>

#include <string>
#include <yaml-cpp/yaml.h>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/recorder.hpp"
#include "tools/trajectory.hpp"

// 定义命令行参数
const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  // 运行模式读自 config 的 hw_buff.run_mode：
  // 1=跟随电控mode(0/1不控制,2小符,3大符) 2=强制小符 3=强制大符；非法值回退为跟随电控
  int run_mode = 1;
  try {
    auto node = YAML::LoadFile(config_path)["hw_buff"];
    if (node && node["run_mode"]) run_mode = node["run_mode"].as<int>();
  } catch (const std::exception & e) {
    tools::logger()->warn("读取 hw_buff.run_mode 失败({})，回退为跟随电控mode", e.what());
  }
  if (run_mode < 1 || run_mode > 3) run_mode = 1;

  // 调试预测时域：与 aimer 的 predict_time 保持一致
  double pred_h = 0.2;
  try {
    auto node = YAML::LoadFile(config_path);
    if (node["predict_time"]) pred_h = node["predict_time"].as<double>();
  } catch (const std::exception &) {
  }

  // 初始化绘图器、录制器、退出器
  tools::Plotter plotter;
  tools::Recorder recorder;
  tools::Exiter exiter;

  // 初始化云台、相机
  io::Gimbal gimbal(config_path);
  io::Camera camera(config_path);

  // 初始化识别器、解算器、追踪器、瞄准器
  auto_buff::Buff_Detector detector(config_path);
  auto_buff::Solver solver(config_path);

  // 大符链路已替换为 big_rune 相位运动引擎（RP-26Rune 迁移），BigTarget 不再依赖
  // HW tracker 的 set_mode 切换——改用双对象 + 引用分派：小符/大符各自持有引擎，
  // 按当前模式把基类引用 target 绑定到对应对象，后续调试代码保持原变量名不动。
  auto_buff::SmallTarget small_target(config_path);
  auto_buff::BigTarget big_target(config_path);
  auto_buff::Aimer aimer(config_path);

  // ===== 运行模式说明（选自 config: hw_buff.run_mode）=====
  // run_mode=1：跟随电控 mode（0/1 不控制，2→小符，3→大符）；
  // run_mode=2：无视电控，强制小符；run_mode=3：无视电控，强制大符。
  // 大符引擎从建模开始（约 100 帧），模式切换/重新进入大符时会自动重新收敛。
  if (run_mode == 2)
    tools::logger()->info("[auto_buff_debug_mpc] 强制小符模式（忽略电控 mode）。");
  else if (run_mode == 3)
    tools::logger()->info("[auto_buff_debug_mpc] 强制大符模式（忽略电控 mode）。");
  else
    tools::logger()->info(
      "[auto_buff_debug_mpc] 跟随电控 mode：2=小符(SMALL_BUFF), 3=大符(BIG_BUFF)，其余不下发控制。");

  cv::Mat img;
  Eigen::Quaterniond q;
  std::chrono::steady_clock::time_point t;

  while (!exiter.exit()) {
    camera.read(img, t);
    q = gimbal.q(t);
    auto gs = gimbal.state();
    // recorder.record(img, q, t);

    // -------------- 打符核心逻辑 --------------

    // 模式门控：
    // run_mode=2/3 → 有效模式直接定为 SMALL/BIG_BUFF；
    // run_mode=1   → 有效模式=电控 mode：2=小符, 3=大符，其余模式只跟踪、不瞄准不开火。
    const auto mode = (run_mode == 2)   ? io::GimbalMode::SMALL_BUFF: (run_mode == 3) ? io::GimbalMode::BIG_BUFF: gimbal.mode();
    // 引用分派：小符走 SmallTarget（HW 链路），大符走 BigTarget（big_rune 相位引擎）。
    // 非打符模式绑 small_target 仅喂空观测，保持状态机推进语义不变。
    auto_buff::Target & target =
      (mode == io::GimbalMode::BIG_BUFF)
        ? static_cast<auto_buff::Target &>(big_target)
        : static_cast<auto_buff::Target &>(small_target);
    const bool buff_mode =
      mode == io::GimbalMode::SMALL_BUFF || mode == io::GimbalMode::BIG_BUFF;

    solver.set_R_gimbal2world(q);

    // ===== 分段计时（调试曲线用）=====
    auto t0 = std::chrono::steady_clock::now();
    auto power_runes = detector.detect(img);
    auto t1 = std::chrono::steady_clock::now();

    solver.solve(power_runes);
    auto t2 = std::chrono::steady_clock::now();

    // 非打符模式喂空观测：tracker 状态机照常推进，15 帧后自动 reset，
    // 重新进入打符模式时从头收敛，不会带着旧状态直接开打。
    target.get_target(buff_mode ? power_runes : std::nullopt, t);
    auto t3 = std::chrono::steady_clock::now();

    auto target_copy = target;

    auto_aim::Plan plan;
    if (buff_mode) plan = aimer.mpc_aim(target_copy, t, gs, true);
    auto t4 = std::chrono::steady_clock::now();

    gimbal.send(
      plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
      plan.pitch_acc);
    // -------------- 调试输出 --------------

    nlohmann::json data;

    // 主循环节拍：上一帧到这一帧的间隔（ms），看帧率/卡顿
    static auto last_loop_t = std::chrono::steady_clock::now();
    auto now_t = std::chrono::steady_clock::now();
    data["loop_dt_ms"] = std::chrono::duration<double, std::milli>(now_t - last_loop_t).count();
    last_loop_t = now_t;

    // 分段耗时（ms）：detect(推理) / solve(PnP+坐标系) / track(EKF) / aim(弹道迭代)
    auto ms = [](auto a, auto b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    data["detect_ms"] = ms(t0, t1);
    data["solve_ms"] = ms(t1, t2);
    data["track_ms"] = ms(t2, t3);
    data["aim_ms"] = ms(t3, t4);

    // buff原始观测数据
    if (power_runes.has_value()) {
      const auto & p = power_runes.value();
      data["buff_R_yaw"] = p.ypd_in_world[0];
      data["buff_R_pitch"] = p.ypd_in_world[1];
      data["buff_R_dis"] = p.ypd_in_world[2];
      data["buff_yaw"] = p.ypr_in_world[0] * 57.3;
      data["buff_pitch"] = p.ypr_in_world[1] * 57.3;
      data["buff_roll"] = p.ypr_in_world[2] * 57.3;
    }

    // 注意：target 可解 != 本帧有检测值（tracker 处于 TEMP_LOST 时仍可解），
    // 因此这里必须同时判断 optional，否则下面 value() 会抛 std::bad_optional_access
    if (buff_mode && !target.is_unsolve() && power_runes.has_value()) {
      auto & p = power_runes.value();

      // 显示
      for (int i = 0; i < 4; i++) tools::draw_point(img, p.target().points[i]);
      tools::draw_point(img, p.target().center, {0, 0, 255}, 3);
      tools::draw_point(img, p.r_center, {0, 0, 255}, 3);

      // 当前帧target更新后buff
      if (mode == io::GimbalMode::BIG_BUFF) {
        // 大符（big_rune 相位引擎）：HW 的 yaw/roll 重投影语义不再适用，
        // 直接重投影引擎输出的世界系打击点。
        // 绿点=当前打击点(dt=0)，蓝点=pred_h 时域预测点；建模收敛后蓝点应稳定领先绿点。
        tools::draw_point(
          img, solver.reproject_point(big_target.engine()->predict_aim_point(0.0)), {0, 255, 0}, 3);
        tools::draw_point(
          img, solver.reproject_point(big_target.engine()->predict_aim_point(pred_h)), {255, 0, 0},
          3);
      } else {
        // 小符（HW 链路）：当前帧姿态重投影
        auto Rxyz_in_world_now = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
        auto image_points =
          solver.reproject_buff(Rxyz_in_world_now, target.ekf_x()[4], target.ekf_x()[5]);
        tools::draw_points(img, image_points, {0, 255, 0});

        // buff瞄准位置(预测)：用 pred_h 时域的预测 roll 重投影，蓝点应稳定领先绿点
        // （注：target 与 target_copy 浅拷贝共享 EKF，二者 ekf_x() 相同，旧的
        //  dangle 恒为 0，故这里改用 predict_roll 的结果画出真正的预测姿态）
        auto Rxyz_in_world_pre = target.point_buff2world(Eigen::Vector3d(0.0, 0.0, 0.0));
        image_points = solver.reproject_buff(
          Rxyz_in_world_pre, target_copy.ekf_x()[4], target_copy.predict_roll(pred_h));
        tools::draw_points(img, image_points, {255, 0, 0});
      }

      // pred_h 时域的预测 roll（小符=HW predictor；大符=相位引擎，roll 语义兼容）
      const double pred_roll = target_copy.predict_roll(pred_h);

      // ===== 预测值单独导出 =====
      // ① 预测打击点球坐标：mpc_aim 弹道迭代最后一次 predict 的结果（飞行时间时域）
      const Eigen::Vector3d aim = target_copy.aim_point_world();
      data["aim_yaw"] = std::atan2(aim.y(), aim.x()) * 57.3;  // 度
      data["aim_pitch"] =
        std::atan2(aim.z(), std::hypot(aim.x(), aim.y())) * 57.3;  // 度，向上为正
      data["aim_dis"] = aim.norm();                                 // m

      // ② 配置时域 pred_h 后的预测 roll / 预测角速度（数值差分）
      data["pred_roll"] = pred_roll * 57.3;                              // 度
      data["pred_spd"] = (target_copy.predict_roll(pred_h + 0.002) - pred_roll) /
                         0.002 * 57.3;  // 度/s

      // ③ 预测超前量：pred_h 时域的 roll 领先当前 EKF roll 多少（度）
      data["pred_lead"] = (pred_roll - target.ekf_x()[5]) * 57.3;

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["R_yaw"] = x[0];
      data["R_V_yaw"] = x[1];
      data["R_pitch"] = x[2];
      data["R_dis"] = x[3];
      data["yaw"] = x[4] * 57.3;

      data["angle"] = x[5] * 57.3;
      data["spd"] = x[6] * 57.3;
      if (x.size() >= 10) {
        data["spd"] = x[6];
        data["a"] = x[7];
        data["w"] = x[8];
        data["fi"] = x[9];
        data["spd0"] = target.spd;
      }
    }

    // ===== 大符相位引擎专属曲线（仅 BIG_BUFF 模式导出，含无检测帧，可看断流区间）=====
    // 相位链：big_raw_phase_a/b（双候选原始观测相位）-> big_cont_phase（连续化）
    // -> big_fitted_phase（模型拟合），收敛后 cont 与 fitted 应重合；
    // 转速链：big_omega/a/b/spd 对应 v(t)=a*cos(ωt+δ)+b 的模型参数与当前值。
    if (mode == io::GimbalMode::BIG_BUFF) {
      const auto & eng = *big_target.engine();
      data["big_raw_phase_a"] = eng.raw_phase_a() * 57.3;        // 主候选原始相位（度）
      data["big_raw_phase_b"] = eng.raw_phase_b() * 57.3;        // 副候选原始相位（度）
      data["big_cont_phase"] = eng.continuous_phase() * 57.3;    // 连续化相位（度）
      data["big_fitted_phase"] = eng.fitted_phase_now() * 57.3;  // 拟合相位（度）
      const auto & m = eng.motion();
      data["big_omega"] = m.speed_angular_frequency;  // ω（rad/s）
      data["big_amp"] = m.speed_amplitude;            // a 振幅（rad/s）
      data["big_bias"] = m.phase_linear_velocity;     // b 基础转速（rad/s）
      data["big_spd"] = eng.speed_now();              // 当前转速 v(t)（rad/s）
      data["big_model_ready"] = eng.model_ready() ? 1 : 0;  // 运动方程有效性
      data["big_radius"] = eng.radius();              // 符心->靶心半径 EMA（m）
    }

    // 云台响应情况
    data["gimbal_yaw"] = gs.yaw * 57.3;
    data["gimbal_pitch"] = gs.pitch * 57.3;
    data["gimbal_yaw_vel"] = gs.yaw_vel * 57.3;
    data["gimbal_pitch_vel"] = gs.pitch_vel * 57.3;

    // tracker 状态机阶段：0=CONVERGING 1=TRACKING 2=TEMP_LOST 3=LOST
    // （全程导出，LOST 时曲线掉到 3，能直接看到跟踪断流的区间）
    // 注：大符模式无 HW 状态机，此值恒为 3(LOST)；断流/有效性看 big_model_ready。
    data["tracker_status"] = static_cast<int>(target.tracker_status());

    // aimer 跳变检测：突变累计次数 / 当前帧是否判定切扇叶（禁火）
    data["mistake_count"] = aimer.mistake_count();
    data["switch_fanblade"] = aimer.switch_fanblade() ? 1 : 0;

    if (plan.control) {
      data["plan_yaw"] = plan.yaw * 57.3;
      data["plan_pitch"] = plan.pitch * 57.3;
      data["plan_yaw_vel"] = plan.yaw_vel * 57.3;
      data["plan_pitch_vel"] = plan.pitch_vel * 57.3;
      data["plan_yaw_acc"] = plan.yaw_acc * 57.3;
      data["plan_pitch_acc"] = plan.pitch_acc * 57.3;
      data["shoot"] = plan.fire ? 1 : 0;
    }

    plotter.plot(data);

    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("result", img);

    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  return 0;
}