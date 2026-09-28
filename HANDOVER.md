# 2027_preview 打符模块交接文档（SZU 检测 + HW 算法链合入）

> 交接日期：2026-09-26（合代码） / 2026-09-27（彻底合入、仓库自包含）
> 交接范围：`tasks/auto_buff` 原模块被替换为合并算法链（**非独立程序**）
> 合入形态：**HW 与 SZU 源码已拷入仓库内，模型已放入 `assets/rune_model/`，仓库自包含、拷走即可编译**

---

## 1. 背景

NUC 为 **Intel Iris Xe 集显（i5-1135G7，Ubuntu 24.04，无 NVIDIA GPU）**，原 `auto_buff` 的 yolo11_buff 检测 + 自研 EKF 打符方案效果不理想。经讨论定案：

- **检测**：SZU `26_NNDeployment_Lib_and_Detection_Models` 的 `RuneModel`（OpenVINO，Intel 集显可跑）
- **PnP / 跟踪 / 预测**：`HWauto_buff2026` 的 `BuffPnPSolver` / `BuffTracker` / `BuffPredictor`
- **弹道**：沿用 2027_preview 自带 `tools::Trajectory`
- **形态**：真正合入 `tasks/auto_buff` 原模块，对外接口不变，调用方零改动
- **2026-09-27 彻底合入**：HW/SZU 源码不再外部链接，全部拷入 `tasks/auto_buff/` 子目录；模型放入 `assets/rune_model/`

---

## 2. 最终算法链

```
图像帧
  │
  ▼
Buff_Detector::detect(img)        ← SZU RuneModel::netProcess（OpenVINO GPU，sync 推理，5 关键点）
  │   输出 PowerRune（像素 4 角点 上/左/下/右 + R 字中心）
  ▼
Solver::solve(power_runes)        ← HW BuffPnPSolver（IPPE，BUFF_WIDTH=0.114m，内置 cv_to_tf）
  │   填充 pose_camera / xyz_in_world / ypr_in_world / R_gimbal2world / t_camera2gimbal
  ▼
SmallTarget / BigTarget::get_target(power_runes, ts)   ← HW BuffTracker push/update + BuffPredictor set_state
  │   内部计算世界系打击点 aim_point_world_
  ▼
Aimer::aim / mpc_aim              ← 两次 Trajectory 迭代弹道闭环（含阻力/重力）
  │   target.predict(dt) → HW BuffPredictor::predict_position(dt) → 世界系打击点
  ▼
io::Command / auto_aim::Plan
```

对外接口签名保持不变：
`Buff_Detector::detect / detect_24 / detect_debug` → `Solver::solve` → `SmallTarget/BigTarget::get_target/predict/point_buff2world/ekf_x/is_unsolve` → `Aimer::aim/mpc_aim`。

---

## 3. 文件改动清单

### 3.1 核心模块（tasks/auto_buff/）
| 文件 | 改动 |
|---|---|
| `buff_detector.hpp/cpp` | 重写：内部改为 SZU `RuneModel`（pimpl 前置声明）；`detect` 取置信度最高结果，5 点转 `FanBlade`（上/左/下/右）+ R 中心；**推理模式 sync** |
| `buff_solver.hpp/cpp` | 重写：`solve` 改用 HW `BuffPnPSolver`；`pnp_solver_` 为 `mutable`；填充 `pose_camera`/`R_gimbal2world`/`t_camera2gimbal`；**R 中心优先用 point_R 反投影求交（射线×扇叶平面），失败回退几何法** |
| `buff_target.hpp/cpp` | 重写：内部改为 HW `BuffTracker` + `BuffPredictor`（shared_ptr，可拷贝）；`get_target` 从 PowerRune 取位姿喂 tracker；`predict(dt)` 返回世界系打击点；`ekf_x()` 兼容旧接口 |
| `buff_type.hpp` | `PowerRune` 增加 `pose_camera`(4x4)、`R_gimbal2world`、`t_camera2gimbal`、`r_center`、`r_center_is_measured` 字段 |
| `buff_aimer.hpp/cpp` | 扇叶跳变抑制优化：只抑制当帧、不重置 fire_gap 计时器（避免 800 帧只发 3~6 发） |
| `CMakeLists.txt` | 源列表移除 `yolo11_buff.cpp`；`add_subdirectory(hw_buff_algo)` + `add_subdirectory(szu_nn)`；链接 `auto_aim openvino::runtime ${CERES_LIBRARIES} NNdeployment_lib buff_algorithm` |

### 3.2 合入的算法子库（2026-09-27 新增，仓库内自包含）
| 目录 | 来源 | 内容 |
|---|---|---|
| `tasks/auto_buff/hw_buff_algo/` | HWauto_buff2026 | include/（buff_algo 全部头）+ src/（locate/predictor/selector/tracker，**detector 剔除**）+ config/（tracker/predictor/detector yaml）；`CMakeLists.txt` 编译 `buff_algorithm` 静态库 |
| `tasks/auto_buff/szu_nn/` | 26_NNDeployment_Lib_and_Detection_Models | interface/ + inc/ + src/（**仅 OpenVINO 路径，TensorRT 剔除**）；`CMakeLists.txt` 编译 `NNdeployment_lib` 静态库 |
| `assets/rune_model/` | SZU 模型 | `Rune-v8n-fp16-20260624-D14367-B16.{xml,bin}`（bin 5.8MB） |

### 3.3 调用方（src/ 与 tests/）
| 文件 | 改动 |
|---|---|
| `src/auto_buff_debug.cpp` | `SmallTarget target;` → `SmallTarget target(config_path);` |
| `src/auto_buff_debug_mpc.cpp` | 同上 |
| `src/standard_mpc.cpp` | 两个 target 构造传 `config_path` |
| `src/uav.cpp` | 同上 |
| `src/mt_standard.cpp` | 同上 |
| `tests/auto_buff_test.cpp` | `BigTarget target(config_path);` |

### 3.4 构建与配置
| 文件 | 改动 |
|---|---|
| 主 `CMakeLists.txt` | **已移除外部 `add_subdirectory(../26_NNDeployment...)` 与 `add_subdirectory(../HWauto_buff2026...)`**；算法子库由 `tasks/auto_buff/CMakeLists.txt` 内部引入 |
| `configs/*.yaml`（sentry/uav/standard3/standard4/demo/mvs/ascento） | `hw_buff.config_dir` → `.../tasks/auto_buff/hw_buff_algo/config`；`rune_model.path` → `.../assets/rune_model/Rune-v8n-fp16-20260624-D14367-B16.xml` |
| `readme.md` | 末尾「打符算法合并说明」 |
| `HANDOVER.md` | 本文档 |

### 3.5 已删除 / 不参与编译的文件
- `src/auto_buff_szu_hw.cpp`、`src/auto_buff_hw.cpp`（独立程序形态，已废弃删除）
- `tasks/auto_buff/yolo11_buff.cpp/hpp`、`buff_predict.hpp`：**仍在目录中但已不参与编译**（死代码，见 §8）
- `tasks/auto_buff.bak_20260926/`：原模块整目录备份（回滚安全网），已加入 `.gitignore`，不进仓库

---

## 4. 关键设计决策

### 4.1 坐标系桥（最重要，勿改）
HW `BuffPnPSolver` **内部已经做了 cv_to_tf 旋转**（四元数 (-0.5,0.5,-0.5,0.5)，等价于 `R_camera2gimbal`），输出的 `pose.translation` 已经在相机**"前 x 左 y 上 z"** 系。

因此 solver 到 gimbal 系 **只加平移**：
```cpp
Eigen::Vector3d xyz_in_gimbal = pose.translation.cast<double>() + t_camera2gimbal_;
```
**绝对不要**再乘 `R_camera2gimbal`——否则双重旋转，y/z 含义全反，R 心、roll、打击点全错（这是之前 bug 的根因，已修）。

到世界系：`xyz_in_world = R_gimbal2world_ * xyz_in_gimbal`（`R_gimbal2world` 由 `set_R_gimbal2world(q)` 每帧更新）。

### 4.2 弹道闭环
`Aimer::get_send_angle` 用 2027 自带 `tools::Trajectory`（RK4，含阻力，重力 9.781）做**两次迭代闭环**：
1. `target.predict(predict_time)` → 取打击点 → `Trajectory0(v0, d, h)` 得 fly_time
2. `target.predict(trajectory0.fly_time)` → 取新打击点 → `Trajectory1` 得 pitch

每次 `predict(dt)` 调用 HW `BuffPredictor::predict_position(dt)`，HW predictor 内部补偿 state age，即"HW predictor 被弹道迭代反复调用"。

### 4.3 时间戳
统一 `std::chrono::steady_clock`，转 double 秒：
```cpp
const double ts = std::chrono::duration<double>(timestamp.time_since_epoch()).count();
```
get_target 里 `predictor_->set_state(state, ts, ts)`（base_ts = state_ts = ts），后续 `predict(dt)` 即预测 dt 秒后位置。

### 4.4 颜色 / 模式
从 config 读（不是从 ECS 串口收）：
- `hw_buff.buff_mode`: 1=小符(SMALL_BUFF), 2=大符(BIG_BUFF) —— 决定构造 SmallTarget 还是 BigTarget
- `hw_buff.buff_color`: 0=蓝方, 1=红方（当前红方）

### 4.5 丢失保护
- 连续丢失 > 15 帧 → `tracker_->reset()` 并置 unsolvable
- 可解判定：`tracker->status() != buff_algo::StatusType::LOST`（降级保护；大符 RANSAC 未 ready 时处于 CONVERGING 仍视为可解，见 §7）
- `Target::spd` 在 `predict()` 中更新（ekf_x 为 const，不能赋值）

### 4.6 已知隐患（已分析，未修）
- **相机系外推**：HW predictor 的 ω 在相机系里估计；实车云台跟随符时 roll_diff≈0 → ω 塌陷 → 提前量失效。修法：喂 tracker 前把位姿乘 `R_gimbal2world` 转世界系（详见讨论记录）
- **±72° 五重模糊**：5 片扇叶对称导致 roll 有 5 个候选；修法：喂 tracker 前枚举 k·72° 取与上一帧最近者

---

## 5. 配置文件

`configs/sentry.yaml`（其它 yaml 同结构）：
```yaml
hw_buff:
  config_dir: "/home/asus/Alan/2027_preview/tasks/auto_buff/hw_buff_algo/config"
  buff_color: 1        # 0=蓝方, 1=红方
  buff_mode: 2         # 1=小符, 2=大符
  confidence_threshold: 0.5
  nms_threshold: 0.45
rune_model:
  path: "/home/asus/Alan/2027_preview/assets/rune_model/Rune-v8n-fp16-20260624-D14367-B16.xml"
```

HW 内部参数（勿随意改，与算法强耦合）：
- `tracker.yaml`：small（switch_buff_angle=45.8°, R_center_filter_ratio=0.1）、big（RANSAC min_inliers=100, omega 1.884~2.0, A 0.78~1.045）、temp_lost_return_frames=5
- `predictor.yaml`：空节点

外参（各 yaml 独立）：`camera_matrix` / `distort_coeffs` / `R_camera2gimbal` / `t_camera2gimbal` / `R_gimbal2imubody` 按各自标定值。

**换机器提醒**：所有路径均为绝对路径（`/home/asus/Alan/2027_preview/...`），换环境需批量替换；后续可改为相对路径。

---

## 6. 构建与运行

```bash
# 构建（仓库自包含，无需外部同级目录）
cmake -B build
make -C build -j$(nproc)

# 离线验证（打符视频；注意视频路径不带扩展名）
./build/auto_buff_test /path/to/符 --config-path=configs/sentry.yaml --start-index=0 --end-index=N

# 实车（示例）
./build/standard configs/sentry.yaml
```

**注意**：
- OpenCV `CommandLineParser` 对 `-c xxx` 短参数解析有 bug（会把 config-path 解析成 "true"），必须用 `--config-path=xxx` 等号形式
- 模型依赖 OpenVINO（系统路径 `/opt/intel/openvino_2024.6.0/`）；`CMakeLists.txt` 中 `OpenVINO_DIR` 需与本机一致

---

## 7. 验证结果

- ✅ 全量 `make -j4` 编译通过（MAKE_EXIT=0，含内部子库 hw_buff_algo/szu_nn）
- ✅ 离线冒烟：`auto_buff_test` 跑 demo 视频正常退出（EXIT=0），RuneModel 从 `assets/rune_model/` 加载成功、tracker/predictor/aimer 链路无崩溃
- ✅ 打符视频实测（SZU `测试视频/符.avi`，200 帧采样）：**detect_hit=82**，检测到后全部可解；全片 800 帧命中 671（前 118 帧为视频开头无目标段）
- ✅ 模型在 Intel Iris Xe GPU 上正常运行（OpenVINO device="GPU"，**sync 推理**）

**待改进（已讨论未定）**：
- 大符 RANSAC 冷启动保护：当前用 `status() != LOST` 降级；更严格方案是给 `BuffTracker` 加 public `ransac_ready()` 转发（需改 hw_buff_algo）
- 相机系外推隐患（§4.6）：实车前必须验证/修复
- ±72° 五重模糊（§4.6）：精确度收益最大，建议先做
- 模型输入分辨率：当前 SZU 默认 640×384；如换模型需同步确认后处理

---

## 8. 遗留项 / 待办

| 项 | 说明 | 建议 |
|---|---|---|
| `yolo11_buff.cpp/hpp`、`buff_predict.hpp` | 死代码，不参与编译 | 确认不再需要后删除 |
| demo.avi 59.75MB | 超过 GitHub 50MB 推荐上限（推送仅 warning） | 移出仓库或 Git LFS |
| 相机系外推 | HW predictor ω 在相机系估计，实车云台跟随会塌陷 | 喂 tracker 前转世界系 + 台架验证 |
| ±72° 五重模糊 | roll 5 候选歧义 | 枚举 k·72° 取与上帧最近 |
| 大符 RANSAC 保护 | 现为降级方案 | 按 §7 加 `ransac_ready()` |
| 模型精度 | Rune-v8n-fp16 已可用 | 有更高精度模型可替换 `rune_model.path` 后复测 |
| 其它 main 的配置 | 已补 6 个 yaml | 新增兵种 main 需同步补 `hw_buff`/`rune_model` 段 |
| 绝对路径 | configs 内为 /home/asus/Alan/... 绝对路径 | 换机器批量替换或改相对路径 |

---

## 9. 回滚方案

原模块完整备份在 `tasks/auto_buff.bak_20260926/`（含 yolo11_buff，可随时还原）：
```bash
# 备份当前合并版
mv tasks/auto_buff tasks/auto_buff.szu_hw
# 还原原版
mv tasks/auto_buff.bak_20260926 tasks/auto_buff
# 还原 CMakeLists 与调用方构造行（见 git 历史）
```

---

## 10. 外部依赖（已合入，仓库自包含）

| 依赖 | 现在的位置 | 说明 |
|---|---|---|
| HW 算法链（PnP/tracker/predictor） | `tasks/auto_buff/hw_buff_algo/` | 源码已拷入仓库，编译 `buff_algorithm`；detector（TensorRT）已剔除 |
| SZU 检测（RuneModel） | `tasks/auto_buff/szu_nn/` | 源码已拷入仓库，编译 `NNdeployment_lib`；仅 OpenVINO 路径 |
| 检测模型 | `assets/rune_model/` | Rune-v8n-fp16 .xml/.bin |
| OpenVINO | 系统安装 | `/opt/intel/openvino_2024.6.0/`，CMakeLists 的 `OpenVINO_DIR` 需匹配 |

**结论**：不再依赖 `../26_NNDeployment...`、`../HWauto_buff2026` 同级目录；仓库单独拷走即可编译（仅需系统有 OpenCV/Eigen3/Ceres/OpenVINO/yaml-cpp 等常规依赖）。
