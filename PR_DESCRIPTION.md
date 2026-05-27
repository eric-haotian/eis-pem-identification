# PR 描述：SEIS 模型优化与物理参数辨识改进

本 Pull Request 对 `eis_pem_identification` 原型引入了重大增强。它扩展了物理参数空间，优化了参数辨识与优化策略，提升了诊断工具的性能，并补充了多项消融与错配分析研究，使整个代码库具备更好的物理实际贴合度和更高的鲁棒性。

## 修改摘要

### 1. 扩展物理真实性（增至 48 个参数）
我们新增了 **7 个物理参数**（总参量从 41 个扩展至 48 个），消除了以往简化模型中不符合物理实际的假设：
- **CPE 双电层导纳**：在 `_dfn_component_impedances()` 中引入了 `alpha_dl_neg` 和 `alpha_dl_pos`，用恒相位元件（CPE）导纳 $C_{dl} \cdot (j\omega)^\alpha$ 替换了理想电容（$j\omega C_{dl}$），从而能够准确描述由多孔电极表面粗糙度引起的阻抗半圆下凹（depressed semicircles）现象。
- **正极 CEI 薄膜阻抗**：移除了原先硬编码的 `Rsei_pos = 0` 和 `Csei_pos = 0` 简化，引入了与负极 SEI 类似的正极电解质界面膜（CEI）阻抗计算：`rou_sei_pos_0`（参考温度下的 CEI 电阻率）和 `epse_sei_pos`（CEI 介电常数）。
- **可调 Bruggeman 指数**：将原先在 `_updated_parameters()` 中硬编码的 `brug_neg = brug_pos = brug_sep = 1.5` 改为可自由优化的参量规格：`brug_neg`、`brug_pos` 和 `brug_sep`，以适配实际电极不同的弯曲度（tortuosity）。

### 2. 模量加权目标函数（Modulus-Weighted Cost）
- 在代价计算及优化流程中引入了 `modulus_weighting: bool` 选项。
- 通过使用 $1 / |Z_{obs}|$ 对残差进行加权（即 Boukamp 模量加权法），有效平衡了高频与低频数据对拟合的影响，防止优化器过度拟合高频部分的随机噪声。

### 3. 多起点全局优化机制（Multi-Start Optimizer）
- 增强了 `LeastSquaresOptimizer`，增加了 `n_starts` 参数。
- 当 `n_starts > 1` 时，优化器会在对数变换空间的参数边界内随机均匀采样多个初始点，并从每个初始点分别运行局部拟合，最终返回最优结果。这极大降低了 48 维高维空间中优化器陷入局部极小值的风险。

### 4. 参量相关性诊断与相关性预筛选
- **诊断分析**：在 `IdentifiabilityReport` 中增加了 `correlation_matrix` 属性，并提供了 `export_correlation_csv` 工具，用于导出由 Fisher 信息矩阵（$J^T J$）计算得出的参数相关性矩阵。
- **稳健选择器预筛选**：在 `IdentifiabilitySelector.select()` 中引入了基于相关性的预筛选步骤。在进行 SVD 消除循环之前，如果任意两个非保护参数的绝对相关系数 $|\rho| > 0.98$，则自动固定敏感度范数较低的参数。该方法成功将稳健子问题的条件数从 $\sim 10^4$ 优化至 $\mathbf{1.13 \times 10^3}$。

### 5. 补充验证性研究与消融测试（Ablation, Mismatch & Stability）
新增了多项专用脚本与单元测试以验证以下关键特性：
- **全电池 vs. 解耦模型消融 (Full-Cell vs. Decoupled)**：对比了仅使用全电池 EIS 数据与使用解耦电极阻抗数据在参数可辨识性（SVD 秩和条件数）上的显著差异。
- **模型错配测试 (Model Mismatch)**：量化评估了当电池中存在接触电阻（`R_contact`）或寄生电感（`L_ind`）但模型中未包含这两项时，物理参数估计的误差偏移。
- **高频截断影响 (High-Frequency Truncation)**：系统性研究了截断高频测量点对寄生电感 `L_ind` 可辨识性的影响。
- **稳健选择稳定性 (Robust Selection Stability)**：验证了在对先验值（Priors）进行不同程度的物理扰动下，稳健选择器输出的可辨识参数集合的稳定性。

---

## 验证与测试结果

### 1. 单元测试
所有单元测试均已成功通过：
```bash
$ PYTHONPATH=src pytest
============================= 25 passed in 35.44s ==============================
```

### 2. 示例脚本验证
- **联合参数辨识基准 (Joint PEM Benchmark)** (`run_joint_all_seis_parameters_pem.py`)：
  - 在解耦阻抗谱和辅助校准参数配合下，成功估计全部 48 项参数。
  - Jacobian 矩阵达到满秩 ($48/48$)，条件数为 $2.42 \times 10^3$，最终参数识别的误差中位数仅为 **0.15%**（最大误差 7.25%）。
- **稳健参数选择基准 (Robust Selection Benchmark)** (`run_robust_all_seis_parameters_pem.py`)：
  - 在全电池工况及 $0.5\%$ 相对复高斯噪声下，稳健选择器自动锁定 21 个自由参数，固定其余 27 个参数。
  - 识别参数的误差中位数仅为 **2.22%**（最大误差 12.40%）。

---

## 文件改动详情

| 组件 / 文件路径 | 变更说明 |
| :--- | :--- |
| [`src/eis_pem/seis_model.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/seis_model.py) | 新增 7 个参数，实现 CPE 导纳、CEI 薄膜阻抗以及可调 Bruggeman 指数计算 |
| [`src/eis_pem/costs.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/costs.py) | 实现了模量加权（Modulus Weighting）机制 |
| [`src/eis_pem/optimizers.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/optimizers.py) | 添加多起点（Multi-start）局部优化循环及模量加权兼容性 |
| [`src/eis_pem/diagnostics.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/diagnostics.py) | 补充参数相关性矩阵计算及 CSV 导出工具 |
| [`src/eis_pem/robust.py`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/src/eis_pem/robust.py) | 添加相关性预筛选步骤，提升 SVD 辨识子问题稳定性 |
| [`tests/`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/tests/) | 更新相关断言、参数边界以及期待变量列表，新增多起点与相关性诊断测试 |
| [`examples/`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/examples/) | 更新了原有的参数辨识脚本，并新增了消融、模型错配、高频截断及稳定性研究脚本 |
| [`README.md`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/README.md) | 同步更新参数计数、模型表述及示例输出说明 |
| [`.gitignore`](file:///Users/zhonghaotian/Desktop/battery/eis_pem_identification/.gitignore) | 配置过滤自动生成的 `data/` 和 `outputs/` 文件夹 |
