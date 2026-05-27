# EIS PEM Identification

这是一个电池 EIS 参数辨识原型：
正向模型、预测误差 cost、优化器和结果诊断彼此分离。项目包含首阶段
Randles 参数恢复，以及直接依据 `SEIS-Toolbox-LIB` 解析公式实现的
DFN-like `Z_cell` 物理参数辨识路径；运行时不要求 MATLAB。

学习AI推荐本人开发的www.haotianblog.com

## Model

Randles-like 模型是 PEM 闭环中的首个正向阻抗模型，而 PEM 是利用该正向模型进行参数反演的优化方法。

```text
Z(w) = Rs + Rct / (1 + j * w * Rct * Cdl)
w = 2 * pi * f
```

辨识参数为 `Rs`、`Rct` 和 `Cdl`。默认 synthetic case 使用：

| Parameter | True value | Bounds | Unit |
| --- | ---: | ---: | --- |
| `Rs` | 0.01 | `1e-4` to `1e-1` | ohm |
| `Rct` | 0.05 | `1e-4` to `1.0` | ohm |
| `Cdl` | 2000.0 | `10` to `1e5` | F |

频率网格为 `np.logspace(-2, 5, 100)`。观测阻抗在真值上加入固定随机种子
生成的相对复高斯噪声，其中实部和虚部噪声的逐点标准差均为
`0.005 * abs(Z_true)`。

## Installation And Run

```bash
python -m pip install -e ".[test]"
python examples/run_synthetic_eis_pem.py
python examples/run_synthetic_seis_pem.py
python examples/run_all_seis_parameters_pem.py
python examples/run_robust_all_seis_parameters_pem.py
python examples/run_joint_all_seis_parameters_pem.py
pytest
```

示例使用 `scipy.optimize.differential_evolution` 在 `log10` 参数空间内优化，
并在终端打印真值、辨识值及相对误差。

## Outputs

运行示例后会生成：

```text
data/synthetic_eis.csv
data/fit_result.csv
outputs/nyquist_fit.png
outputs/bode_magnitude_fit.png
outputs/bode_phase_fit.png
outputs/residuals.png
```

`synthetic_eis.csv` 列定义为：

```text
freq_Hz,Zreal_ohm,Zimag_ohm,Zreal_true_ohm,Zimag_true_ohm
```

`fit_result.csv` 列定义为：

```text
freq_Hz,Zreal_obs_ohm,Zimag_obs_ohm,Zreal_fit_ohm,Zimag_fit_ohm,Zreal_true_ohm,Zimag_true_ohm,residual_real_ohm,residual_imag_ohm
```

## SEIS Physical Parameter Identification

`SEISModel` 将上游 MATLAB 工具箱的 `Parameters_update`、
`Model_particle_calculate` 与 `Model_DFN_calculate` 的 `Z_cell` 计算路径
移植到 Python。返回值是面积比阻抗，单位为 `ohm*m^2`。

当前同时辨识工具箱移交说明和论文敏感度脚本指定的全部四项负极物理参数：

| Parameter | Toolbox default | Bounds | Unit |
| --- | ---: | ---: | --- |
| `Ds_neg_0` | `1.2e-14` | `1e-15` to `1e-13` | `m^2/s` |
| `rs_neg` | `2.0e-6` | `0.5e-6` to `10e-6` | `m` |
| `k_neg_0` | `5.031e-11` | `1e-12` to `1e-9` | reaction-rate parameter |
| `rou_sei_neg_0` | `1.4025e5` | `1e4` to `1e6` | `ohm*m` |

`run_synthetic_seis_pem.py` 在 `298.15 K`、`SOC=1.0`、`0.5%` 相对复噪声
条件下生成 synthetic `Z_cell`，用 relative PEM cost 在 log-space 中同步
辨识以上四项参数。它生成：

```text
data/synthetic_seis.csv
data/seis_fit_result.csv
data/seis_identified_parameters.csv
outputs/seis_nyquist_fit.png
outputs/seis_bode_magnitude_fit.png
outputs/seis_bode_phase_fit.png
outputs/seis_residuals.png
```

### All Scalar Physical Inputs

`run_all_seis_parameters_pem.py` 提供完整参数化运行路径。它将上游
`Parameters_initialize.m` 中实际进入 SEIS 公式的全部 `48` 项独立标量
物理输入同时作为优化变量，包括正负极反应、扩散、几何、孔隙率、导电率、
电解液、SEI、化学计量端点和 Arrhenius 活化能参数。

为使温度活化能和 SOC 相关参数进入观测，完整运行使用 `5` 个温度与 `5`
个 SOC 组成的 `25` 工况 synthetic 实验，每工况 `60` 个频率点。它输出：

```text
data/synthetic_all_seis_experiments.csv
data/all_seis_fit_result.csv
data/all_seis_identified_parameters.csv
data/all_seis_identifiability.csv
data/all_seis_singular_values.csv
```

该运行保留 `F`、`R`、参考温度 `T_0` 作为物理常量，并保留文献给定的 OCP
函数形式；温度与 SOC 是实验条件而不是电池待辨识参数。局部 Jacobian
诊断必须与参数结果一起解读：完整参数化问题即使在多工况、无噪声
synthetic 数据上也可能高度病态，因此近零残差不等于全部参数均具有唯一、
抗噪声的物理识别结果。

### Robust All-Parameter Workflow

`run_robust_all_seis_parameters_pem.py` 在相同的 `25` 工况设计上加入 `0.5%`
相对复高斯噪声，并使用分阶段可辨识性选择，而不是将 `48` 项全部声明为
可独立辨识：

1. 在 relative-residual、参数变换空间中计算局部 Jacobian 与奇异值；
2. 固定弱奇异方向主导的非保护参数，直至自由子问题条件数不超过 `1e4`；
3. 在假定 `0.5%` 噪声下继续固定预测 `95%` 相对区间超过 `10%` 的非保护参数；
4. 始终保留 `Ds_neg_0`、`rs_neg`、`k_neg_0` 与 `rou_sei_neg_0` 作为保护目标，
   若其预测区间超限则以警告状态报告而不是静默固定。

完整 `48` 项参数仍写入结果文件；状态为 `fixed_identifiability` 的项目使用
工具箱名义值作为显式假设，其值不被表述为 EIS 独立恢复结果。稳健运行生成：

```text
data/robust_all_seis_fit_result.csv
data/robust_all_seis_parameters.csv
data/robust_all_seis_selection.csv
data/robust_all_seis_singular_values.csv
outputs/robust_all_seis_parameter_uncertainty.png
outputs/robust_all_seis_singular_values.png
outputs/robust_all_seis_residuals.png
```

### Joint Identification Of All 48 Parameters

`run_joint_all_seis_parameters_pem.py` 提供一个所有 `48` 项均作为自由变量的
synthetic 闭环。该流程不声称全电池单通道 EIS 能单独支撑全部参数，而是将
工具箱 `Model_DFN_calculate.m` 已定义的区域解耦谱 `Z_neg`、`Z_pos`、
`Z_sep` 与独立物性校准读数共同纳入同一个 relative PEM 目标函数。

流程先以区域谱诊断 EIS-only 问题，自动确定仍需独立校准的信息。本 benchmark
中需校准 `12` 项参数，每项模拟 `9` 次 `0.5%` 相对噪声读数；随后全部
`48` 项进入同一个 least-squares 自由向量，不再固定任何参数。输出包括：

```text
data/joint_decoupled_seis_observations.csv
data/joint_all_seis_fit_result.csv
data/joint_all_seis_parameters.csv
data/joint_all_seis_auxiliary_measurements.csv
data/joint_decoupled_eis_only_selection.csv
data/joint_all_seis_identifiability.csv
data/joint_all_seis_singular_values.csv
outputs/joint_all_seis_singular_values.png
outputs/joint_all_seis_residuals.png
```

该闭环证明在“解耦频谱 + 明确记录的辅助物性数据”条件下可稳定估计全部参数。
它不等价于仅从普通全电池 EIS 或尚未提供的真实实验数据中恢复全部物理参数。

## Model Contract

`ForwardModel` 的固定接口为：

```python
simulate(freq_hz: np.ndarray, theta: np.ndarray) -> np.ndarray
```

`RandlesModel` 与 `SEISModel` 都遵守该接口并复用同一套 PEM cost、
optimizer 与诊断输出。DEIS 及 GCD/SEIS/DEIS 联合辨识仍不属于当前范围。
