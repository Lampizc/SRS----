# README_coords — 坐标链与波束域约定（全项目唯一权威文档）

> **地位**：本文与 `srs_track/contract.py` 第 2 节是同一套约定的两种载体。
> 二者冲突 = bug，先改契约、跑自检，再同步本文。
> **适用**：`scene.py` / `array.py` / `obs.py`（同学 A）、`est.py` / `crb.py`（同学 B）、
> `group.py`（同学 C）、以及一切要画指向图、对拍方向误差的人。
> **改参数**：只改 `config.yaml`（唯一真源），改完跑 §6 的三行验证。
> **改动本文或契约任何符号前**，先跑自检，全 PASS 才许合入。

---

## 0. 四级坐标链一图流

```
 (1) 几何方向        (2) 方向余弦         (3) 空间频率            (4) 波束索引
 角度 α, φ    ──►    u, v          ──►    ψ_u, ψ_v         ──►    k_u, k_v
                     u = sinα·cosφ         ψ = (d/λ_ul)·u          k = N·ψ（连续）
                     v = sinα·sinφ         上行波长！               rint(k) = 最近 DFT 格点
                     u²+v² ≤ 1（圆盘）     d/λ_ul ≈ 0.4532         k ∈ [−N/2, N/2)
```

**接口里传的"角度"永远是 (u, v) 方向余弦**（契约的 `theta_uv` / `theta_hat` /
`theta_pred` 全是 (K, 2) 的 (u, v)）。α、φ 只允许出现在场景生成与展示层，
一进接口就必须已经是 (u, v)。四人之间不传 α、φ。

---

## 1. 第 1 级：几何角 → 方向余弦（u, v）

用户方向用相对阵列法向（boresight）的**离轴角 α** 与**方位角 φ** 表示：

```
u = sin(α)·cos(φ)        v = sin(α)·sin(φ)
```

三条必须记住的约束：

1. **单位圆盘，不是整球面**：u² + v² = sin²α ≤ 1。(u=0.9, v=0.9) 这种组合
   物理上不存在。生成用户位置必须**先在圆盘上采样再转角度**，禁止对 u、v
   独立均匀采样（那会采出方形盘，中心密度还错）。
2. 参考配置下服务区边缘 α_max ≈ 52.3°，故实际取值范围是 u² + v² ≤ 0.6265，
   比单位圆盘更小。
3. |u| 的物理上界 = sin(α_max) ≈ 0.7915（假设阵列法向指向星下点）。

## 2. 第 2 级：方向余弦 → 空间频率 ψ

```
ψ_u = (d / λ_ul) · u        ψ_v = (d / λ_ul) · v
```

1. **必须用上行波长 λ_ul**：SRS 是上行信号，物理传播与采样都发生在上行。
   只有在重建下行预编码流形时才换 λ_dl（见 §7 FDD 提醒）。
2. d/λ_ul ≈ 0.4532 < 0.5，所以可见区 |u| ≤ 1 映射到 |ψ| ≤ 0.4532，严格落在
   DFT 的一个周期 (−0.5, 0.5) 内 → **无空间混叠 → 波束索引唯一，不必处理环绕**。
   这是本模型敢直接用"波束索引"的物理依据（契约 `__post_init__` 会拦住
   d/λ 超出 (0, 0.5] 的配置）。

## 3. 第 3 级：空间频率 → 波束索引 k

```
k = N · ψ          （连续浮点值）
```

1. `psi_to_beam_index` 返回**连续值**，这是有意的：估计的"离格"（off-grid）
   本质就是 k 不是整数，残差 `k − rint(k)` 是局部参数拟合要用的信息。
   **不要提前取整**，取整只发生在两处：选观测窗中心（`make_window` 内部）和
   报告"落在哪个波束"。
2. 合法范围 k ∈ [−N/2, N/2)，N=64 即 [−32, 31]。
3. 数组下标换算（写切片/查数组时用）：
   - **FFT 下标 = k mod N**（计算用，`beam_index_to_fft_index`），如 k=−6 → 58；
   - **fftshift 下标 = k + N/2**（画图用，`beam_index_to_shifted_index`）。
4. 参考配置的量级：|k| ≤ N·(d/λ_ul)·sin(α_max) ≈ 22.96，加观测窗半宽 3 仍
   < N/2 = 32 → 窗不会越过 DFT 周期边界（契约自检"观测窗不越界"盯着；
   换小阵列调试时这条会真的触发，见 §9）。

## 4. 第 4 级：阵列流形与 DFT（符号约定，全项目只此一种）

```
单元相位（+j 上升）： a_u[m] = exp(+j·2π·ψ_u·m),  m = 0 … N−1
二维平铺：            a = kron(a_u, a_v)，平铺下标 i = m_u·N_v + m_v（C 序）
酉 DFT 的列：         F[k, m] = (1/√N)·exp(−j·2π·k·m/N)  ——  恰是 conj(a)，即接收权
波束域变换：          Y_beam = fft2(y_2d) / sqrt(N_u·N_v)   （标准正变换！不是 ifft）
```

1. 方向 ψ = k/N 的流形在 DFT 下的峰值**恰好落在第 k 个波束**。
2. `to_beamspace` 的除 √N_ant 是**酉归一化，不可省**：它保证 ‖Y‖² = ‖y‖²
   （能量守恒），进而"单模式噪声能量 = 阵元噪声方差 σ²"成立。
   去掉它，所有 SNR 差 N_ant 倍。
3. 平铺顺序 `i = m_u·N_v + m_v` 必须与 `steering_upa` 的 reshape 一致
   （两者都在契约里，别在自己模块里另写一份平铺）。
4. **符号写反的后果**：把 a 写成 exp(−j…) 或用 ifft，峰值跑到 k = −Nψ，
   表现为角度整体镜像。误差大小是对的、方向全错，只看误差曲线发现不了，
   只能靠自检的"波束峰值位置"一项守。

## 5. SNR 口径（跟坐标链一起背下来）

```
阵列级、单符号、上行：  SNR_array = |g|²·N_ant / σ²
反解噪声方差：          σ² = |g|²·N_ant / 10^(SNR_dB/10)     （noise_var_from_array_snr）
```

1. **σ² 是每阵元的复噪声总方差 E|n|²**；按实/虚两维拆时每维 σ²/2，生成噪声
   写 `n = sqrt(σ²/2)·(randn + j·randn)`。误按"每维 σ²"生成会让实际 SNR
   比标称低 3 dB，且误差曲线上看不出来（2026-11-05 修订过契约里歧义表述）。
2. SNR 按**单符号**定义；L 个符号相干合并后有效 SNR = SNR + 10log10(L)。
3. 只保留 B 个波束时，窗内信号能量因泄漏减少，噪声只剩 B·σ²——**不能**
   直接拿 SNR_array 当窗内 SNR，做 CRB 时按实际窗能量比换算。
4. 报 SNR 必须带全三个限定词：**阵列级、单符号、上行**。少一个，结果可能差 20~30 dB。

## 6. 三行验证（每次改坐标相关代码后跑）

```bash
# 第 1 行：契约自检（10 项，必须全 PASS）
python -m srs_track.contract
```

```python
# 第 2、3 行：方向 (0.12, -0.20) 的波束域峰值必须落在 rint(N·ψ) —— 符号约定人工抽查
import numpy as np
from srs_track.contract import SystemConfig, steering_upa, to_beamspace, make_window
cfg = SystemConfig(); uv = np.array([0.12, -0.20])
w = make_window(uv, cfg, 3); Y = to_beamspace(steering_upa(uv, cfg), w, cfg)
peak = (int(w.k_u[np.argmax(np.abs(Y))]), int(w.k_v[np.argmax(np.abs(Y))]))
assert peak == tuple(np.rint(cfg.n_u * cfg.d_over_lambda_ul * uv).astype(int)), peak
print("峰值波束 OK:", peak)          # 期望 (3, -6)
```

改了参数（config.yaml）之后额外跑：

```bash
python -m srs_track.config    # config.yaml → 契约对象往返校验 + 派生量打印
```

## 7. FDD 提醒：几何互易，流形不互易

- FDD 上下行载波不同（1.995 / 2.185 GHz），**几何方向 (u, v) 互易，
  阵列流形不互易**（相位随波长缩放）。
- 用上行 SRS 估出几何角 (u, v) 后，重建**下行**流形必须显式传下行间距比：

  ```python
  a_dl = steering_upa(uv, cfg, d_over_lambda=cfg.d_over_lambda_dl)   # ≈ 0.4963
  ```

- 忘传这个参数会默认用 d/λ_ul，造成约 **9%** 的下行波束指向偏差——
  这正是申报书要求"按下行波长重建阵列响应"的原因。

## 8. 参考数值表（config.yaml 参考配置下的派生量）

| 量 | 值 | 说明 |
|---|---|---|
| λ_ul | 0.1502719 m | C_LIGHT / f_ul_hz |
| λ_dl | 0.1372048 m | C_LIGHT / f_dl_hz |
| d/λ_ul | 0.453179 | 坐标链缩放因子（< 0.5 → 无混叠） |
| d/λ_dl | 0.496338 | 下行流形重建用（≈ 半波长） |
| DFT 格距 Δψ | 1/64 = 0.015625 | 一个"模式间隔"（申报书窗半宽的单位） |
| 对应 Δu | 0.034479 | 格距折算到方向余弦域 |
| 服务区地面半径 | 853.5 km | 球模型，min_elev_deg = 30° |
| 最大离轴角 α_max | 52.325° | 服务区边缘相对星下点 |
| \|u\| 上界 sin(α_max) | 0.79149 | u²+v² ≤ 0.6265 |
| \|k\| 上界 | 22.956 | 加半宽 3 仍 < 32，窗不越界 |
| σ² @ SNR 20/30/40 dB | 40.96 / 4.096 / 0.4096 | E\|n\|²（\|g\|=1，每实数维再除 2） |

改了 config.yaml 后本表以 `python -m srs_track.config` 打印的为准。

## 9. 常见错误对照表（症状 → 根因 → 防线）

| 症状 | 根因 | 防线 |
|---|---|---|
| 角度整体镜像（误差大小对、方向反） | 流形写成 exp(−j…) 或用了 ifft | 自检"波束峰值位置" |
| 所有 SNR 差 N_ant 倍 | to_beamspace 忘了 /√N_ant | 自检"波束域酉归一化" |
| 所有 SNR 差 3 dB | 噪声按"每实数维 σ²"生成 | §5 口径 + obs.py 单测（SNR 往返） |
| 用户出现在物理上不存在的方向（如 u=v=0.9） | 对 u、v 独立采样而非圆盘采样 | make_window 的 u²+v² 检查 |
| 下行波束指向偏 ~9% | 下行流形用了默认 d/λ_ul | §7：显式传 d_over_lambda_dl |
| 单用户形状退化成 (2,) | 违反铁律 1 | 契约形状守卫（会当场 raise） |
| 小阵列（如 8×8）调试时窗越界报错 | \|k\|+半宽 超过 N/2 | make_window 越界检查（这是保护，不是 bug） |
| 参数改了结果没变 / 四人参数不一致 | 绕过 config.yaml 硬编码 | 每周自查 + `python -m srs_track.config` |

## 10. 函数速查（实现都在 contract.py；array.py 正式化时逐位一致，不得另写）

| 函数 | 作用 |
|---|---|
| `uv_to_psi(uv, cfg)` | (u,v) → ψ（上行；只做线性缩放，不取模） |
| `psi_to_beam_index(psi, n)` | ψ → 连续波束索引 k = N·ψ（不取整） |
| `beam_index_to_fft_index(k, n)` | k ∈ [−N/2, N/2) → FFT 数组下标 [0, n) |
| `beam_index_to_shifted_index(k, n)` | k → fftshift 后下标（画图用） |
| `steering_upa(uv, cfg, d_over_lambda=None)` | UPA 流形；下行重建显式传 d_over_lambda_dl |
| `to_beamspace(y, window, cfg)` | 阵元域 → 有限 DFT 波束域（酉归一化） |
| `make_window(uv_center, cfg, half_width_modes)` | 以**预测**方向为中心造观测窗 |
| `noise_var_from_array_snr(snr_db, cfg, gain_abs)` | 阵列级 SNR → σ²（E\|n\|²） |

---

*维护：同学 A（模块 1）。本文随契约版本更新；当前对应 contract 1.0.1（2026-11-05）。*
