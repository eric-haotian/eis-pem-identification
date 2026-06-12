# EIS-PEM Adaptive Parameter Identification System: GUI User Manual

> Learn AI at [www.haotianblog.com](https://www.haotianblog.com)

This manual explains how to use the **graphical user interface (GUI)** application for EIS-PEM parameter identification.

This tool uses CSV files as input and Excel workbooks (`.xlsx`) as output. The basic workflow is simple:

1. Prepare a `.csv` file containing the test data.
2. Double-click or run the program, select the input file, and click **Run**.
3. Open the automatically generated `.xlsx` file to review all identified physical parameters.

---

## 1. Prepare the Input Data (`input.csv`)

You need to organize your experimental data into a standard comma-separated values file (`.csv`). You may use Excel to prepare the table, then select **Save As → CSV (Comma delimited)**.

The input table **must contain only the following five column headers**, and the spelling must match exactly:

| Column Name | Description |
|---|---|
| `Frequency_Hz` | Frequency in hertz (Hz) |
| `Re_Z` | Real part of impedance, in ohms (Ω) |
| `Im_Z` | Imaginary part of impedance, in ohms (Ω) |
| `Temperature_K` | Absolute temperature in kelvin. For example, 25°C = 298.15 K |
| `SOC` | Battery state of charge, expressed as a decimal between 0 and 1 |

> [!TIP]
> **How do I perform multi-condition joint fitting? This is required.**
>
> The software can process spectra measured at multiple temperatures and SOC values at the same time. You only need to vertically concatenate all spectra into a single CSV file, one after another, and specify the corresponding `Temperature_K` and `SOC` values for each row.
>
> The system will automatically read the combined table and perform high-dimensional physics-based joint decoupling.
>
> At least **3 different temperatures** and **3 different SOC values** are required. Using **5 or more** different conditions is recommended for better identification performance.

---

## 2. Run the GUI Main Program

Make sure Python is installed on your computer. Then enter the project directory:

```bash
cd your_installation_path/eis_pem_identification
```

Run the GUI program:

```bash
python eis_pem_gui.py
```

A clean graphical interface will appear. The GUI contains three main sections:

### 2.1 File Selection

Click the **Browse...** button and select the `.csv` data file you prepared.

### 2.2 Parameter Configuration

Configure the following options before running the identification:

| Option | Description |
|---|---|
| **Expected Noise Level** | Expected noise ratio. Typical values are `0.01` for 1% noise or `0.03` for 3% noise. If the data are noisy, the system will automatically activate the physics-based data cleaner to filter noise. For larger datasets, the program becomes more conservative, so this value should be adjusted appropriately. |
| **Random Starts** | Number of multi-start optimization attempts. Values such as `3` or `5` are usually sufficient. This helps prevent the optimizer from becoming trapped in poor local minima. |
| **Error Strictness (`max_ci95`)** | Error tolerance threshold. A larger value means the system allows higher uncertainty. |

### 2.3 Run and Log Output

Click **▶ RUN IDENTIFICATION**.

The black log window will print the full workflow in real time, including data cleaning, SVD-based dimensionality diagnostics, and algorithm convergence.

---

## 3. Understand the Excel Output (`_Results.xlsx`)

When the log window shows `Success!` and a success dialog appears, the program automatically generates a formatted `_Results.xlsx` file in the same directory as the selected `.csv` file.

Open the Excel workbook. It contains two sheets.

---

### Sheet 1: `Summary_Diagnostics`

This sheet records the overall fitting status, including metrics such as the total root mean square error (RMSE).

---

### Sheet 2: `Parameters`

This is the core result sheet.

It lists all **48 microscopic physical parameters** row by row. The main columns are:

| Column | Description |
|---|---|
| `Parameter` | Parameter name, such as solid-phase diffusion coefficient, porosity, and other physical quantities |
| `Initial/Reference` | Built-in physical reference value used by the system |
| `Fitted_Value` | Actual parameter value inferred by the system |
| `CI95_Absolute` | Absolute error at 95% confidence |
| `Status` | Identification status of the parameter |

The `Status` column may contain the following values:

| Status | Meaning |
|---|---|
| `Fitted` | The parameter was successfully identified. This means the input data contain enough information to independently decouple this parameter. |
| `Fixed` | The parameter was frozen. This is expected when the current data do not contain enough information to support independent identification of that parameter, either because the noise level is too high or because the dataset lacks sufficient multi-condition support. For scientific rigor, the system avoids unsupported parameter guessing. |

---

## 4. Notes on Model Parameters and Example Data

The optimized `SEISModel` contains **48 independent scalar physical inputs** as optimization variables.

For the complete parameter list, see:

```text
ALL_Parameters.md
```

The recommended test dataset is:

```text
Input_Spectra_25Conditions.csv
```

Use this example dataset to verify that the GUI and identification workflow are functioning correctly before processing your own experimental data.

---

## 5. Recommended Workflow

1. Prepare the CSV file with the required five columns.
2. Ensure the dataset includes at least 3 different temperatures and 3 different SOC values.
3. Run `eis_pem_gui.py`.
4. Select the input CSV file in the GUI.
5. Set the expected noise level, random starts, and error strictness.
6. Click **▶ RUN IDENTIFICATION**.
7. Open the generated `_Results.xlsx` file.
8. Check `Summary_Diagnostics` first, then inspect the detailed parameter results in `Parameters`.

---

## 6. Output Interpretation Summary

A successful result does not necessarily mean all 48 parameters will be marked as `Fitted`.

In practice:

- `Fitted` means the parameter is independently identifiable from the current data.
- `Fixed` means the software intentionally kept the parameter fixed because the data do not provide enough independent information.

This behavior is designed to improve physical reliability and prevent overfitting.

For best results, use clean spectra collected across multiple temperatures and SOC values.

# Then introduction in Chinese

# EIS-PEM 参数自适应鉴别系统：图形化使用手册 (GUI)
学习AI就上www.haotianblog.com
这篇手册将教你如何使**图形化界面应用程序 (GUI)**。

本工具完全基于 CSV 表格输入和 Excel (XLSX) 表格输出，你只需要：
1. 准备好包含测试数据的 `.csv` 表格。
2. 双击运行程序，用鼠标选择文件并点击“Run”。
3. 打开自动生成的 `.xlsx` 文件查看所有物理参数。

---

## 1. 准备输入数据 (`input.csv`)

你需要将实验数据整理为一个标准的逗号分隔值表格 (`.csv`) 文件。你可以使用 Excel 整理数据，然后点击“另存为 -> CSV (逗号分隔)”。

你的表格**必须且仅需**包含以下五个表头（请完全匹配拼写）：
* `Frequency_Hz`: 频率（赫兹）
* `Re_Z`: 阻抗实部（欧姆）
* `Im_Z`: 阻抗虚部（欧姆）
* `Temperature_K`: 绝对温度（开尔文，例如 25°C = 298.15 K）
* `SOC`: 电池荷电状态（0 到 1 之间的小数）

> [!TIP]
> **如何进行多工况联合拟合？这个是必须项！**
> 本软件可以同时处理多个温度和 SOC 的谱图。你只需要把所有谱图的数据**首尾相连垂直拼在同一个 CSV 表格里**，并在对应的 `Temperature_K` 和 `SOC` 列中标明它们属于哪个工况即可。系统会自动读取并进行高维物理联合解耦。
> 至少3个不同温度和不同SOC，推荐5个及以上，这样效果更好。

---

## 2. 运行图形化主程序

确保你的电脑安装了 Python 后，进入代码目录，运行我们为你准备好的主程序：
```bash
cd 你的安装路径/eis_pem_identification
```
```bash
python eis_pem_gui.py
```

此时会弹出一个非常清爽的图形化窗口，分为三个部分：
1. **文件选择区**：点击 `Browse...` 按钮，选择你刚刚准备好的 `.csv` 数据文件。
2. **参数配置区**：
   - **Expected Noise Level**: 预期噪声比例。通常填 `0.01` (1%) 或 `0.03` (3%)。如果你的数据很脏，系统会自动开启物理清洗器过滤杂音。数据越大越程序保守，建议调整至合适位置。
   - **Random Starts**: 多起点搜索次数。填 `3` 或 `5` 或别的整数即可，保证优化器不会卡在死胡同里。
   - **Error Strictness (max_ci95)** 误差容忍度，数字越大容忍度越高。
3. **运行与日志区**：点击 **▶ RUN IDENTIFICATION** 按钮。你会看到下方的黑色日志框里实时打印出系统清洗数据、SVD 降维诊断以及算法收敛的全过程。

---

## 3. 如何看懂 Excel 输出结果 (`_Results.xlsx`)

当黑色日志框提示 `Success!` 并在屏幕弹出成功提示弹窗后，你选择的那个 `.csv` 文件的同目录下会自动生成一个格式排版好的 `_Results.xlsx` Excel 文件。

打开这个 Excel 文件，里面有两张 Sheet：

### Sheet 1: `Summary_Diagnostics`
这里记录了整体的拟合状态，比如总均方根误差 (RMSE)。

### Sheet 2: `Parameters` (核心结果)
这里逐行列出了所有 48 个微观物理参数。你会看到以下列：
* **`Parameter`**: 参数名（如固相扩散系数、孔隙率等）。
* **`Initial/Reference`**: 系统自带的物理基准初值。
* **`Fitted_Value`**: 系统推算出的实际参数值。
* **`CI95_Absolute`**: 95% 置信度的绝对误差。
* **`Status`**: 该参数的状态。
  - **`Fitted` (成功识别)**：恭喜，你的数据足以独立解耦这个参数！
  - **`Fixed` (被冻结)**：不要紧张！这意味着系统诊断出当前的数据不足以支撑该参数的独立识别（要么因为噪声太大，要么因为缺乏多工况支撑）。为了你的科研严谨性，系统拒绝瞎猜。

在优化的SEISModel 中，共有 48 项 独立标量物理输入作为优化变量。在ALL_Parameters.md文件中。
Input_Spectra_25Conditions.csv 为推荐测试用数据集。
