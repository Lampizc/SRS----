# -*- coding: utf-8 -*-
"""
srs_track.contract — 项目接口契约
================================================================================

项目：面向低轨卫星通信的 SRS 角度跟踪与资源调度研究
版本：CONTRACT_VERSION = "1.0.1"

--------------------------------------------------------------------------------
这个文件是干什么的
--------------------------------------------------------------------------------
本文件不含任何算法，它只做一件事：把所有会互相传的东西，提前用一个可运行的数据结构固定下来。
算法实现分别落在 scene.py / obs.py / est.py / crb.py / group.py / predict.py / budget.py / sched.py / metrics.py
但那些文件之间只允许通过本文件定义的类型通信。

若某个模块目录被加到 sys.path 首位，`import types` 会被本文件顶掉，numpy 等库内部导入会随机报错，且报错位置与真实原因完全无关，极难排查。

--------------------------------------------------------------------------------
模块归属（谁实现哪个 Protocol）
--------------------------------------------------------------------------------

    代号     角色           主责模块                              实现的文件
    ----     ----           --------                              ----------
    同学 A   队长           场景与观测建模 / 平台集成             scene.py
                                                                  array.py
                                                                  obs.py
    同学 B   估计理论       角度估计 / Fisher 信息与 CRB          est.py
                                                                  crb.py
    同学 C   分组算法       候选筛选 / 整组精度检查 / 复杂度      group.py
    同学 D   调度与实验     预测器 / 误差预算 / 调度 / 统计       predict.py
                                                                  budget.py
                                                                  sched.py
                                                                  metrics.py

    Protocol 对照：SceneGenerator / Observer       → 同学 A
                   Estimator                       → 同学 B
                   Grouper                         → 同学 C
                   Predictor / Scheduler / Metrics → 同学 D


每人在自己的文件里实现对应的 Protocol 类即可，不需要改本文件。

--------------------------------------------------------------------------------
使用方式
--------------------------------------------------------------------------------
    from srs_track.contract import SystemConfig, Scene, Observation, Estimate
    # 或在本目录下： from contract import SystemConfig

自检（验证本契约内部是否自洽，任何时候都应全部 PASS）：

    python -m srs_track.contract
    # 或 python contract.py

自检失败意味着契约本身有矛盾，必须在动手写算法之前解决，不要让四个人在错误的约定上各写几百行。

--------------------------------------------------------------------------------
三条铁律
--------------------------------------------------------------------------------
1. 形状永不含糊：方向向量一律 (K, 2)，K 个用户两列 (u, v)。哪怕只有 1 个用户
   也必须是 (1, 2)，绝不能退化成 (2,)。这是跟踪类代码最高频的 bug 来源。
2. 真值与可观测量严格隔离：TruthTable 只有仿真器和评测代码能读，算法模块
   （est / group / sched）拿到的是 Observation，里面没有真值。违反这条，所有
   精度结果全部作废，而且从曲线上看不出来。
3. 随机性只有一个入口：所有随机数必须由 Scene.seed 派生，禁止在算法内部直接
   调用 np.random.*。否则"5 个独立随机种子可复现"这一考核要求无法满足。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import NDArray

# 契约版本。任何人改动结构都必须升这个号，并在组会记录里写明改了什么。
#
# v1.0.1（2026-11-05）：
#   1. 新增 Predictor Protocol（归同学 D）。此前 theta_pred / cov_pred 被
#      Observer 与 Grouper 消费，却没有任何协议负责生产——"上一拍估计到
#      下一拍预测"这段闭环胶水处于无人认领状态，M2 一跑就会暴露。
#   2. Estimate 新增可选字段 cov（(K,2,2) 估计协方差）；缺省时预测器按
#      Σ ≈ (crb_radius²/2)·I 做各向同性近似。
#   3. 形状守卫自检补充 cov 的检查。
#   4. 文档修订（2026-11-05，不改任何接口，版本不升）：统一 σ² 的口径表述——
#      σ² 是每阵元复噪声总方差 E|n|²（每实数维 σ²/2），原文"方差 σ²（每维）"
#      会被读成每实数维方差，让 obs.py 把实际 SNR 做低 3 dB；另修正文件头
#      版本注释仍写 1.0.0 的笔误。
CONTRACT_VERSION = "1.0.1"

# 光速，单位 m/s。用精确值而非 3e8，避免波长换算出现万分级偏差。
C_LIGHT = 299_792_458.0

# 地球半径（球模型），单位 m。轨道几何、服务区半径、仰角约束都用它。
EARTH_RADIUS_M = 6_371_000.0


# ==============================================================================
# 第 1 节  系统配置
# ==============================================================================


@dataclass(frozen=True)
class SystemConfig:
    """全项目唯一的物理与阵列参数。

    这个对象必须被**所有**模块共享同一份实例，由 Scene.config 传下去。任何
    模块都不得自己硬编码 600km / 64 / 1.995GHz 之类的常数。

    冻结（frozen=True）是有意的：参数一旦下发就不该被中途修改。要跑不同参数，
    新建一个 SystemConfig 实例。

    属性
    ----
    n_ant : int
        天线总数 N_u × N_v。注意这是 4096，不是硬件上的 4096 个射频通道——
        本项目的阵列按全数字阵列处理，若后续要考虑混合波束成形，需要另加模型。
    """

    # --- 轨道与几何 ---
    orbit_height_km: float = 600.0        # 轨道高度
    min_elev_deg: float = 30.0            # 最低服务仰角，决定服务区边界

    # --- 阵列 ---
    n_u: int = 64                         # UPA 第一轴阵元数
    n_v: int = 64                         # UPA 第二轴阵元数
    spacing_m: float = 0.0681             # 阵元间距，约等于下行半波长

    # --- 载波 ---
    f_ul_hz: float = 1.995e9              # 上行（SRS 所在方向）
    f_dl_hz: float = 2.185e9              # 下行（预编码所在方向）

    # --- 时间 ---
    slot_duration_s: float = 0.5e-3       # 一个时隙时长，5G NR 30kHz 子载波间隔
                                          # （0.5ms）。全项目的"时刻"都以时隙索引
                                          # 计，不用秒，避免浮点累积误差。

    # ------------------------------------------------------------------
    # 派生量。全部用 property 计算，不存字段——避免出现"配置改了但派生量
    # 还是旧值"的不一致。
    # ------------------------------------------------------------------

    @property
    def n_ant(self) -> int:
        """天线总数。"""
        return self.n_u * self.n_v

    @property
    def lambda_ul(self) -> float:
        """上行波长，单位 m。约 0.15027 m。"""
        return C_LIGHT / self.f_ul_hz

    @property
    def lambda_dl(self) -> float:
        """下行波长，单位 m。约 0.13720 m。"""
        return C_LIGHT / self.f_dl_hz

    @property
    def d_over_lambda_ul(self) -> float:
        """上行归一化阵元间距 d/λ_ul，约 0.4532。

        这是全项目最重要的一个无量纲数：空间频率 ψ = (d/λ)·u，波束索引
        k = N·ψ。它 < 0.5 保证了可见区（|u| ≤ 1）无空间混叠，因此波束索引
        在 [-N/2, N/2) 内唯一，不需要处理环绕歧义。
        """
        return self.spacing_m / self.lambda_ul

    @property
    def d_over_lambda_dl(self) -> float:
        """下行归一化阵元间距 d/λ_dl，约 0.4963，非常接近半波长。

        FDD 下几何角互易、阵列流形不互易：必须用上行估出的几何角 (u,v)，
        乘以下行 d/λ 重建下行流形。用错这个数会造成约 9% 的波束指向偏差。
        """
        return self.spacing_m / self.lambda_dl

    @property
    def service_radius_m(self) -> float:
        """服务区地面半径（球模型，m）。

        由卫星、地心、用户构成的三角形用正弦定理推出：
            地心角 γ = arccos(Re·cosε / (Re+h)) - ε
        参考配置下约 850 km。
        """
        re = EARTH_RADIUS_M
        rs = re + self.orbit_height_km * 1e3
        eps = math.radians(self.min_elev_deg)
        return re * (math.acos(re * math.cos(eps) / rs) - eps)

    @property
    def max_off_nadir_deg(self) -> float:
        """服务区边缘相对星下点的最大离轴角，单位度。

        由正弦定理：sin(α) = Re·cos(ε) / (Re+h)。参考配置下约 52.3°。
        这个量决定了服务区内 |u| 的上界（sin α ≈ 0.79），进而决定了波束索引
        的取值范围，是判断观测窗会不会越界出 DFT 周期的依据。
        """
        re = EARTH_RADIUS_M
        rs = re + self.orbit_height_km * 1e3
        eps = math.radians(self.min_elev_deg)
        return math.degrees(math.asin(re * math.cos(eps) / rs))

    def __post_init__(self) -> None:
        if self.n_u < 2 or self.n_v < 2:
            raise ValueError("阵列每轴至少 2 个阵元，否则无法形成波束")
        if not (0.0 < self.d_over_lambda_ul <= 0.5):
            raise ValueError(
                f"d/λ_ul = {self.d_over_lambda_ul:.4f} 必须落在 (0, 0.5] 内，"
                "否则可见区会出现空间混叠，波束索引不再唯一"
            )
        if not (0.0 < self.d_over_lambda_dl <= 0.5):
            raise ValueError(f"d/λ_dl = {self.d_over_lambda_dl:.4f} 超出 (0, 0.5]")
        if self.f_ul_hz >= self.f_dl_hz:
            raise ValueError(
                "上行载波应低于下行载波（本项目 1.995 < 2.185 GHz），"
                "若参数确实要反过来，请同步修改申报书中的波长相关表述"
            )


# ==============================================================================
# 第 2 节  坐标与波束域约定（全项目最容易出错的地方，务必逐字读完）
# ==============================================================================
#
# 坐标链条共四级，每一级都必须只有一种写法：
#
#   (1) 几何方向  →  (2) 方向余弦  →  (3) 空间频率  →  (4) 波束索引
#       角度 α,φ        u, v               ψ_u, ψ_v          k_u, k_v
#
# ------------------------------------------------------------------------------
# (1) 方向余弦
# ------------------------------------------------------------------------------
# 用户方向用相对阵列法向（boresight）的离轴角 α 和方位角 φ 表示：
#
#     u = sin(α)·cos(φ)        v = sin(α)·sin(φ)
#
# 关键约束：u² + v² = sin²α ≤ 1。这是一个**单位圆盘**，不是整球面。
# 因此 (u=0.9, v=0.9) 这种组合在物理上不存在，任何随机生成用户位置的代码都
# 必须先按圆盘采样再转角度，不能对 u、v 独立均匀采样。
#
# 参考配置下服务区边缘 α_max ≈ 52.3°，故 u² + v² ≤ 0.63，实际取值空间更小。
#
# ------------------------------------------------------------------------------
# (2) 空间频率
# ------------------------------------------------------------------------------
#     ψ_u = (d / λ_ul) · u        ψ_v = (d / λ_ul) · v
#
# 注意用**上行**波长：SRS 是上行信号，物理传播与采样都发生在上行。
# 只有在重建下行预编码流形时才换成 λ_dl。
#
# 因为 d/λ_ul ≈ 0.4532 < 0.5，可见区 |u| ≤ 1 映射到 |ψ_u| ≤ 0.4532，严格落在
# DFT 的一个周期 (-0.5, 0.5) 内 → 无混叠。这是本模型可以放心用"波束索引"而
# 不必处理环绕的物理依据。
#
# ------------------------------------------------------------------------------
# (3) 波束索引
# ------------------------------------------------------------------------------
# DFT 格点间距为 1/N（沿该轴阵元数的倒数），故连续波束索引：
#
#     k_u = N_u · ψ_u            （连续值，取整后即最近 DFT 格点）
#
# k 的物理范围取 [-N/2, N/2)。与 FFT 数组下标的关系见 beam_index_to_fft_index。
#
# ------------------------------------------------------------------------------
# (4) 阵列流形与 DFT（符号约定，只此一种）
# ------------------------------------------------------------------------------
# 单元相位采用 +j 上升：
#
#     a_u[m] = exp(+j·2π·ψ_u·m),        m = 0 … N_u-1
#     a     = kron(a_u, a_v)            平铺下标 i = m_u·N_v + m_v
#
# DFT 矩阵（酉形式）的列正好是**共轭**流形，也就是指向该波束的接收权：
#
#     F[k, m] = (1/√N)·exp(-j·2π·k·m/N)
#
# 于是波束 k 对应方向 ψ = k/N，且流形 a(ψ=k/N) 在 F 下的峰值恰在 k 处。
# 工程实现上是**标准正变换 FFT** 再加 1/√N_ant 归一化：
#
#     Y_beam = fft2(y_2d) / sqrt(N_u·N_v)        # 酉变换，能量守恒
#
# 若把 a 写成 exp(-j…) 或用 ifft，峰值会跑到 k = -Nψ，表现为"角度符号整体
# 反了"。这个 bug 在只看误差曲线时几乎发现不了（误差大小是对的，只是方向
# 镜像），必须用自检里的 test_beam_peak 来守。改动本节任何符号前，先跑自检。


def uv_to_psi(uv: NDArray, cfg: SystemConfig) -> NDArray:
    """方向余弦 → 归一化空间频率（上行）。

    参数
    ----
    uv : ndarray, shape (K, 2)
        (u, v) 方向余弦。
    cfg : SystemConfig

    返回
    ----
    ndarray, shape (K, 2)
        (ψ_u, ψ_v)。

    说明
    ----
    本函数只做线性缩放，不做任何取模。是否环绕由调用方按 beam 索引处理，
    详见 psi_to_beam_index。
    """
    uv = np.asarray(uv, dtype=float)
    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError(f"uv 形状必须是 (K, 2)，收到 {uv.shape}")
    return uv * cfg.d_over_lambda_ul


def psi_to_beam_index(psi: NDArray, n: int) -> NDArray:
    """空间频率 → **连续**波束索引 k = n·ψ。

    返回浮点值，调用方自行决定是否 np.rint 取整。保留连续值是有意的：
    角度估计的"离格"（off-grid）本质就是 k 不是整数，残差 k - rint(k) 是
    局部参数拟合要利用的信息。
    """
    return np.asarray(psi, dtype=float) * float(n)


def beam_index_to_fft_index(k: int, n: int) -> int:
    """物理波束索引 k ∈ [-n/2, n/2) → FFT 输出数组下标 ∈ [0, n)。"""
    return int(k) % int(n)


def beam_index_to_shifted_index(k: int, n: int) -> int:
    """物理波束索引 k → fftshift 之后的数组下标 ∈ [0, n)。

    绘图、切片时用 fftshift 顺序更直观（负索引在左），但计算时建议仍走
    beam_index_to_fft_index，少一次搬移。
    """
    return int(k) + int(n) // 2


@dataclass(frozen=True)
class BeamWindow:
    """保留的观测波束集合（有限 DFT 观测）。

    这是"有限观测"这一物理约束的载体。申报书要求观测窗半宽取 1、2、3 个
    模式间隔，对应 3×3、5×5、7×7 个波束。

    为什么要强调"至少 2 个波束"：文献 [4] 证明 DFT 波束域中识别单个源至少
    需要 2 个 DFT 波束，窗太窄时角度不可辨识或产生系统偏差。半宽 0 时本类会
    拒绝构造，防止有人在调试时无意中做出"单波束求角"这种在数学上不成立的实验。

    属性
    ----
    k_u, k_v : ndarray, shape (B,)
        B 个波束在两轴上的物理索引，一一对应。**顺序即 Y 的行顺序**，
        不要假设它是按 k 升序排列的。
    n_beams : int
        B = (2·half_width_modes + 1)²。
    """

    k_u: NDArray
    k_v: NDArray
    half_width_modes: int

    def __post_init__(self) -> None:
        ku = np.asarray(self.k_u, dtype=int).ravel()
        kv = np.asarray(self.k_v, dtype=int).ravel()
        if ku.shape != kv.shape:
            raise ValueError(f"k_u 与 k_v 长度不一致：{ku.shape} vs {kv.shape}")
        if ku.size == 0:
            raise ValueError("观测窗不能为空")
        if self.half_width_modes < 1:
            raise ValueError(
                "观测窗半宽至少 1 个模式间隔（即至少 2 个波束）。"
                "文献证明单波束无法辨识角度，见 DFT 波束域 ESPRIT 的可辨识性结论。"
            )
        object.__setattr__(self, "k_u", ku)
        object.__setattr__(self, "k_v", kv)

    @property
    def n_beams(self) -> int:
        """保留的波束数 B。半宽 1/2/3 分别对应 9/25/49。"""
        return int(self.k_u.size)

    @property
    def pairs(self) -> NDArray:
        """(B, 2) 的 (k_u, k_v) 数组，便于一次性传给实现方。"""
        return np.stack([self.k_u, self.k_v], axis=1)

    @property
    def bandwidth_modes(self) -> float:
        """窗宽（模式间隔数）。CRB 对窗宽敏感，做实验曲线时要按这个量分组。"""
        return 2.0 * self.half_width_modes + 1.0


def make_window(
    uv_center: NDArray,
    cfg: SystemConfig,
    half_width_modes: int,
) -> BeamWindow:
    """以预测方向为中心，构造矩形 DFT 观测窗。

    参数
    ----
    uv_center : array_like, shape (2,)
        预测方向 (u, v)。**必须传预测值而不是真值**——现实里只能拿到预测。
    cfg : SystemConfig
    half_width_modes : int
        半宽，单位"模式间隔"。申报书中扫描 1、2、3。

    返回
    -------
    BeamWindow

    说明
    ----
    窗中心取 k = rint(N·ψ)。因为 |k| ≤ 0.453·N 且半宽 ≤ 3，参考配置下
    |k| + 半宽 ≤ 22.9 + 3 = 25.9 < N/2 = 32，窗不会越过 DFT 周期边界。
    若换成更小的阵列（例如调试用的 8×8）这个裕量会变小，届时必须检查是否
    越界——越界后窗口内的波束实际指向会跳变，是隐蔽 bug。
    """
    uv = np.asarray(uv_center, dtype=float).ravel()
    if uv.shape != (2,):
        raise ValueError(f"uv_center 必须是 (2,)，收到 {uv.shape}")
    if float(np.sum(uv ** 2)) > 1.0 + 1e-9:
        raise ValueError(f"uv_center={uv} 不满足 u²+v² ≤ 1，不是合法方向余弦")

    # 两轴同尺寸是当前配置的前提；若未来做非方阵，这里需要分别处理。
    if cfg.n_u != cfg.n_v:
        raise NotImplementedError("当前契约假定方阵 UPA；非方阵需扩展本函数")

    n = cfg.n_u
    psi = uv_to_psi(uv[None, :], cfg)[0]           # (2,)
    k_c = np.rint(psi_to_beam_index(psi, n)).astype(int)
    ku_c, kv_c = int(k_c[0]), int(k_c[1])
    offs = np.arange(-half_width_modes, half_width_modes + 1)

    # 越界保护：物理上服务区不会越界，但调试用小阵列时可能发生。
    ku_min, ku_max = ku_c - half_width_modes, ku_c + half_width_modes
    kv_min, kv_max = kv_c - half_width_modes, kv_c + half_width_modes
    if ku_min < -n // 2 or ku_max > n // 2 - 1:
        raise ValueError(
            f"观测窗在 u 轴越界：[{ku_min},{ku_max}] 超出 [{-n // 2},{n // 2 - 1}]。"
            "缩小半宽或检查阵列尺寸。"
        )
    if kv_min < -n // 2 or kv_max > n // 2 - 1:
        raise ValueError(
            f"观测窗在 v 轴越界：[{kv_min},{kv_max}] 超出 [{-n // 2},{n // 2 - 1}]。"
            "缩小半宽或检查阵列尺寸。"
        )

    # 生成笛卡尔积并展平。顺序固定为 (k_u 变化慢, k_v 变化快)，
    # 与 numpy.meshgrid 的 'ij' 索引一致，Y 的行顺序以此为准。
    KU, KV = np.meshgrid(ku_c + offs, kv_c + offs, indexing="ij")
    return BeamWindow(k_u=KU.ravel(), k_v=KV.ravel(), half_width_modes=half_width_modes)


def steering_upa(
    uv: NDArray,
    cfg: SystemConfig,
    d_over_lambda: float | None = None,
) -> NDArray:
    """二维 UPA 阵列流形（列向量）。

    参数
    ----
    uv : ndarray, shape (2,) 或 (K, 2)
    cfg : SystemConfig
    d_over_lambda : float, optional
        默认用上行 d/λ。**重建下行流形时必须显式传 cfg.d_over_lambda_dl**，
        否则 FDD 场景会引入约 9% 的指向偏差（这正是申报书要求"按下行波长
        重建阵列响应"的原因）。

    返回
    ----
    ndarray, shape (N_ant,) 或 (N_ant, K)
        复数流形。单元相位 a[m] = exp(+j2πψ m)，平铺顺序 i = m_u·N_v + m_v。

    性质
    ----
    ||a||² = N_ant（每个元素模为 1）。这个平方关系在 SNR 定义里会用到。
    """
    uv_in = np.asarray(uv, dtype=float)
    was_1d = uv_in.ndim == 1          # 记录输入是否单用户，决定返回是否降维
    uv2 = np.atleast_2d(uv_in)
    if uv2.shape[1] != 2:
        raise ValueError(f"uv 最后维必须是 2，收到 {uv2.shape}")

    ratio = cfg.d_over_lambda_ul if d_over_lambda is None else float(d_over_lambda)

    m_u = np.arange(cfg.n_u)
    m_v = np.arange(cfg.n_v)

    # psi 已经是 d/λ 归一化后的量；ratio 覆盖时按比例缩放。
    psi = uv2 * ratio
    a_u = np.exp(1j * 2.0 * np.pi * np.outer(m_u, psi[:, 0]))   # (N_u, K)
    a_v = np.exp(1j * 2.0 * np.pi * np.outer(m_v, psi[:, 1]))   # (N_v, K)

    # kron(a_u[:,k], a_v[:,k]) 等价于外积再展平（行优先，i = m_u·N_v + m_v）
    a = (a_u[:, None, :] * a_v[None, :, :]).reshape(cfg.n_ant, uv2.shape[0])

    return a[:, 0] if was_1d else a


def to_beamspace(
    y_elem: NDArray,
    window: BeamWindow,
    cfg: SystemConfig,
) -> NDArray:
    """阵元域观测 → 有限 DFT 波束域观测。

    参数
    ----
    y_elem : ndarray, shape (N_ant,) 或 (N_ant, L)
        单个快拍的阵元域接收，或 L 个符号的接收矩阵。
    window : BeamWindow
        保留的波束集合。
    cfg : SystemConfig

    返回
    ----
    ndarray, shape (B,) 或 (B, L)
        波束域观测，行顺序与 window 的波束顺序严格一致。

    实现要点
    --------
    用 fft2 而不是显式构造 F 矩阵：64×64 变成 4096×4096 的复矩阵是 268 MB，
    B 个波束只需一次 FFT 再取行。此处的除 √N_ant 是酉归一化，保证
    ||Y_beam||² = ||y_elem||²，从而"单模式噪声能量 = 单元噪声方差 σ²"这一
    约定（第 3 节）成立。**去掉这个归一化会让所有 SNR 差 N_ant 倍。**
    """
    y = np.asarray(y_elem)
    squeeze = y.ndim == 1
    if squeeze:
        y = y[:, None]

    if y.shape[0] != cfg.n_ant:
        raise ValueError(f"阵元维应为 {cfg.n_ant}，收到 {y.shape[0]}")

    n_l = y.shape[1]
    y2d = y.reshape(cfg.n_u, cfg.n_v, n_l)

    # 沿两个阵列轴做 DFT；axis=2 是符号维，不动。
    Y_full = np.fft.fft2(y2d, axes=(0, 1)) / np.sqrt(cfg.n_ant)

    iu = np.mod(window.k_u, cfg.n_u)
    iv = np.mod(window.k_v, cfg.n_v)
    Y = Y_full[iu, iv, :]                      # (B, L)

    return Y[:, 0] if squeeze else Y


# ==============================================================================
# 第 3 节  SNR 定义（四人报数对不上，99% 是这个没统一）
# ==============================================================================
#
# 申报书口径：**单符号阵列级信噪比**，定义为
#
#     匹配处理后全模式信号总能量 / 单模式噪声能量
#
# 逐项落实：
#   · 阵元噪声：每阵元独立复高斯，**复噪声总方差** E|n|² = σ²（即每个实数维
#     σ²/2；生成时用 sqrt(σ²/2)·(randn + j·randn)）。经酉 DFT 后每模式仍是 σ²。
#   · 信号在全模式上的总能量：|g|² · ||a||² · |s|² = |g|² · N_ant（取 |s|=1）。
#   · 故 SNR_array = |g|² · N_ant / σ²，反解 σ² = |g|² · N_ant / SNR_array。
#
# 三个必须记住的推论：
#   1. SNR 是按**单符号**定义的。L 个符号相干合并后有效 SNR 变为 L·SNR，
#      角度标准差按 1/√L 下降。这是"增加符号数换取更长更新周期"的物理根源，
#      也是模块 4 里 L/T 权衡的地基。
#   2. 只保留 B 个波束时，窗内信号能量因泄漏而少于全模式总能量，而噪声能量
#      只剩 B·σ²。因此**不能**直接用 SNR_array 当窗内 SNR，必须按实际窗能量
#      比换算。做 CRB 时尤其注意。
#   3. 所有报 SNR 的地方都要注明"阵列级、单符号、上行"。少一个限定词，
#      结果可能差 20~30 dB。

def noise_var_from_array_snr(
    snr_db: float,
    cfg: SystemConfig,
    gain_abs: float = 1.0,
) -> float:
    """由阵列级单符号 SNR 反解阵元噪声方差 σ²。

    参数
    ----
    snr_db : float
        阵列级、单符号 SNR，单位 dB。申报书扫描 20 / 30 / 40 dB。
    cfg : SystemConfig
    gain_abs : float
        用户复增益的幅度 |g|。默认 1（归一化）。

    返回
    ----
    float
        每阵元的**复噪声总方差** E|n|² = σ²。注意不是"每实数维方差"——
        若按实/虚两维拆分，每维是 σ²/2；生成噪声应写
        n = sqrt(σ²/2)·(randn + j·randn)。误按"每维 σ²"生成会让实际
        阵列 SNR 比标称低 3 dB，且误差曲线上看不出来。

    公式
    ----
    σ² = |g|² · N_ant / 10^(SNR_dB/10)
    """
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    return (gain_abs ** 2) * cfg.n_ant / snr_lin


def array_snr_from_noise(
    noise_var: float,
    cfg: SystemConfig,
    gain_abs: float = 1.0,
) -> float:
    """noise_var_from_array_snr 的逆运算，返回 dB。用于自检与日志回读。"""
    return 10.0 * math.log10((gain_abs ** 2) * cfg.n_ant / float(noise_var))


def snr_after_symbols(snr_db: float, n_symbols: int) -> float:
    """L 个符号相干合并后的有效 SNR（dB）。

    角度估计的 CRB 与有效 SNR 成反比，所以这条式子是"符号数—精度"关系的
    最短表达。注意它只在符号间相位已知（已做定时与多普勒预补偿）时成立，
    残余频偏会破坏相干性——那是后续要单独加的失效条件。
    """
    if n_symbols < 1:
        raise ValueError("符号数至少为 1")
    return float(snr_db) + 10.0 * math.log10(n_symbols)


# ==============================================================================
# 第 4 节  数据结构
# ==============================================================================


class UserType(str, Enum):
    """终端移动类型。决定预测误差的增长速率，是模块 4 误差预算的输入。"""

    STATIC = "static"          # 静止：速率 0，预测误差只来自姿态/星历慢漂
    PEDESTRIAN = "pedestrian"  # 行人
    VEHICLE = "vehicle"        # 车辆


# 各类型的典型速率范围，单位 m/s。仅用于场景生成时的抽样，
# 真正进入误差预算的是每个用户的具体速度，不是这个范围。
USER_SPEED_RANGE_MS: dict[UserType, tuple[float, float]] = {
    UserType.STATIC: (0.0, 0.0),
    UserType.PEDESTRIAN: (0.5, 2.0),
    UserType.VEHICLE: (5.0, 30.0),
}


class EstimateStatus(str, Enum):
    """单次角度估计的状态。模块 4 依据它决定是否触发重捕。"""

    OK = "ok"                # 正常，可直接写入跟踪结果
    AMBIGUOUS = "ambiguous"  # 可辨识性不足（如两用户过近），结果不可信
    FAILED = "failed"        # 数值失败（拟合不收敛、样本不足）
    LOST = "lost"            # 与预测偏离过大，判定跟踪失锁


@dataclass(frozen=True)
class UserState:
    """单个地面终端的**可观测**状态（不含真值）。

    真值在 TruthTable 里，二者严格分开，见文件头"三条铁律"第 2 条。
    """

    uid: int
    user_type: UserType
    pos_ecef_m: NDArray      # (3,) 地心地固坐标，m
    vel_ecef_ms: NDArray     # (3,) 速度，m/s
    access_slot: int         # 进入服务区的时隙索引（批量进入场景用）
    exit_slot: int | None = None   # 离开时隙；None 表示整段过境都在区内

    def __post_init__(self) -> None:
        if np.shape(self.pos_ecef_m) != (3,):
            raise ValueError(f"pos_ecef_m 必须是 (3,)，收到 {np.shape(self.pos_ecef_m)}")
        if np.shape(self.vel_ecef_ms) != (3,):
            raise ValueError(f"vel_ecef_ms 必须是 (3,)，收到 {np.shape(self.vel_ecef_ms)}")
        if self.exit_slot is not None and self.exit_slot <= self.access_slot:
            raise ValueError("exit_slot 必须晚于 access_slot")


@dataclass(frozen=True)
class TruthTable:
    """真值表。**仅仿真器与评测代码**可访问。

    为什么单独做一张表而不是塞进 UserState：因为一旦真值能随手拿到，
    算法模块里"不小心用了真值"的错误几乎不可能靠 code review 发现，而它
    会让精度结果好得离谱且完全不可复现。物理隔离是唯一可靠的办法。

    使用约定
    --------
    · scene.py / obs.py（仿真侧）：可以读。
    · metrics.py（评测侧）：可以读，但只用于算误差，不得回灌给算法。
    · est.py / crb.py / group.py / sched.py（算法侧）：**禁止 import 本类**。
      建议在 CI 或组会前 grep 一次这几个文件里有没有 "truth" 字样。
    """

    theta_uv: dict[int, NDArray]   # uid -> (2,) 真值方向余弦
    gain: dict[int, complex]       # uid -> 复增益

    def subset(self, uids: Sequence[int]) -> "TruthTable":
        """取子集，供小规模实验与单用户调试使用。"""
        uids = list(uids)
        return TruthTable(
            theta_uv={u: self.theta_uv[u] for u in uids},
            gain={u: self.gain[u] for u in uids},
        )


@dataclass(frozen=True)
class SatelliteTrack:
    """卫星过境轨迹（星历预测值）。

    只存轨迹不存轨道根数：模块 1 负责把两行根数/星历算成轨迹，其他模块
    一律按"给定时隙返回卫星位置速度"的接口使用，不关心怎么算出来的。
    """

    slots: NDArray        # (T,) 时隙索引，单调递增
    pos_ecef_m: NDArray   # (T, 3)
    vel_ecef_ms: NDArray  # (T, 3)

    # 以下两项是**固定偏差**的来源，模块 4 的误差预算需要，模块 1 负责生成。
    att_err_rad: float = 0.0   # 姿态误差（阵列法向指向偏差），rad
    eph_err_m: float = 0.0     # 星历误差（位置偏差），m

    def __post_init__(self) -> None:
        t = np.shape(self.slots)[0]
        if np.shape(self.pos_ecef_m) != (t, 3):
            raise ValueError(f"pos_ecef_m 应为 ({t}, 3)，收到 {np.shape(self.pos_ecef_m)}")
        if np.shape(self.vel_ecef_ms) != (t, 3):
            raise ValueError(f"vel_ecef_ms 应为 ({t}, 3)，收到 {np.shape(self.vel_ecef_ms)}")
        if np.any(np.diff(self.slots) <= 0):
            raise ValueError("slots 必须严格单调递增")


@dataclass(frozen=True)
class Scene:
    """一个完整的仿真场景 = 配置 + 轨迹 + 用户 + 真值。

    三类布局（uniform / hotspot / corridor）与三类移动类型在 SceneGenerator
    里组合产生。**同一次对照实验的所有方案必须共用同一个 Scene 实例**，
    否则"相同用户集合"这一考核前提不成立。
    """

    config: SystemConfig
    sat: SatelliteTrack
    users: tuple[UserState, ...]
    truth: TruthTable
    layout: str        # "uniform" | "hotspot" | "corridor"
    seed: int          # 本场景随机种子；算法内所有随机性都必须由它派生

    def __post_init__(self) -> None:
        uids = [u.uid for u in self.users]
        if len(uids) != len(set(uids)):
            raise ValueError("用户 uid 重复")
        missing = set(uids) - set(self.truth.theta_uv)
        if missing:
            raise ValueError(f"以下用户缺少真值：{sorted(missing)[:5]}…")

    @property
    def n_users(self) -> int:
        return len(self.users)


@dataclass(frozen=True)
class SRSConfig:
    """单次 SRS 资源配置。

    这是"资源"侧的完整描述。模块 1 负责按 3GPP TS 38.211/38.214 把这里的
    参数映射成实际时频码资源，避免仿真与标准脱节（申报书的已知风险项）。
    """

    n_symbols: int                 # 本次测量占用的 OFDM 符号数 L，>= 1
    comb: int = 2                  # 梳状间隔（comb-2 / comb-4）
    bandwidth_prb: int = 96        # 带宽，PRB 数
    cyclic_shift: int = 0          # 循环移位索引，用于码域正交
    n_ports: int = 1               # 端口数
    hopping: bool = False          # 是否跳频

    def __post_init__(self) -> None:
        if self.n_symbols < 1:
            raise ValueError("n_symbols 至少为 1")
        if self.comb not in (2, 4):
            raise ValueError("comb 取 2 或 4（5G NR 支持值）")
        if self.bandwidth_prb <= 0:
            raise ValueError("bandwidth_prb 必须为正")


@dataclass(frozen=True)
class Observation:
    """算法模块能看到的**全部**输入。没有真值，没有其他后门。

    这是模块 1（同学 A）→ 模块 2（同学 B）的唯一交付物。

    字段
    ----
    theta_pred : ndarray, shape (K, 2)
        预测方向，来自上一时刻估计 + 运动模型外推。**不是真值**。
        它是算法唯一允许使用的先验，用来选观测窗、做局部拟合初值。
    cov_pred : ndarray, shape (K, 2, 2) 或 None
        预测方向的不确定度协方差，供分组前复检与误差预算使用。
        若暂未实现预测模块，置 None，但实现方必须显式处理这种情况。

    形状提醒
    --------
    Y **永远是二维** (B, L)。只要 1 个符号（L=1）也要传 (B, 1)，不能传 (B,)。
    这是有意为之：单快拍与多快拍的代码路径应当完全一致，否则"L 从 1 加到 4"
    这种参数扫描会走进一条从未验证过的分支。to_beamspace 对一维输入会返回
    一维，所以从它取结果时要按需要补一维：Y[:, None]。
    """

    slot: int
    uids: tuple[int, ...]
    window: BeamWindow
    Y: NDArray                      # (B, L) complex，波束域观测
    srs: SRSConfig
    theta_pred: NDArray             # (K, 2)
    cov_pred: NDArray | None = None

    def __post_init__(self) -> None:
        k = len(self.uids)
        y = np.asarray(self.Y)
        if y.ndim != 2:
            raise ValueError(f"Y 必须是 (B, L) 二维数组，收到 {y.shape}")
        if y.shape[0] != self.window.n_beams:
            raise ValueError(
                f"Y 首维 {y.shape[0]} 与窗内波束数 {self.window.n_beams} 不一致"
            )
        if y.shape[1] != self.srs.n_symbols:
            raise ValueError(
                f"Y 次维 {y.shape[1]} 与 n_symbols {self.srs.n_symbols} 不一致"
            )
        if np.shape(self.theta_pred) != (k, 2):
            raise ValueError(
                f"theta_pred 应为 ({k}, 2)，收到 {np.shape(self.theta_pred)}。"
                "注意单用户时也要保持二维 (1, 2)。"
            )
        if self.cov_pred is not None and np.shape(self.cov_pred) != (k, 2, 2):
            raise ValueError(f"cov_pred 应为 ({k}, 2, 2) 或 None")


@dataclass(frozen=True)
class Estimate:
    """模块 2（同学 B）的输出。

    crb_radius 的口径必须全项目统一，否则"测量误差限"这一项在误差预算里
    无法与其他量相加。本契约定义：

        crb_radius = sqrt(trace(CRB)) = sqrt(E[e_u² + e_v²])

    即**误差半径的均方根（RMS）**，不是某个轴的方差，也不是 99% 分位。
    与 99% 分位的换算见 radius_to_99pct。

    theta_hat 与 crb_radius 都是逐用户的，顺序与 uids 一致。

    cov 是可选的估计协方差 (K, 2, 2)。估计器若本来就产出了协方差就填；
    给不出则置 None，Predictor 按各向同性近似换算：

        各向同性时 crb_radius = sqrt(trace(CRB)) = σ·√2
        → Σ_iso ≈ (crb_radius² / 2) · I

    这个换算只供预测外推使用，不改变 crb_radius 的口径本身。
    """

    slot: int
    uids: tuple[int, ...]
    theta_hat: NDArray          # (K, 2)
    crb_radius: NDArray         # (K,) RMS 误差半径下界
    status: EstimateStatus
    group_id: int | None = None
    cov: NDArray | None = None  # (K, 2, 2) 估计协方差，可选

    def __post_init__(self) -> None:
        k = len(self.uids)
        if np.shape(self.theta_hat) != (k, 2):
            raise ValueError(f"theta_hat 应为 ({k}, 2)，收到 {np.shape(self.theta_hat)}")
        if np.shape(self.crb_radius) != (k,):
            raise ValueError(f"crb_radius 应为 ({k},)，收到 {np.shape(self.crb_radius)}")
        if np.any(np.asarray(self.crb_radius) < 0):
            raise ValueError("crb_radius 不能为负")
        if self.cov is not None and np.shape(self.cov) != (k, 2, 2):
            raise ValueError(f"cov 应为 ({k}, 2, 2) 或 None，收到 {np.shape(self.cov)}")


# 二维高斯误差的 99% 分位半径 / RMS 半径。
# 推导：e_u, e_v ~ N(0, σ²) 独立同分布时，r = sqrt(e_u²+e_v²) 服从
# Rayleigh(σ)。RMS(r) = σ√2；99% 分位 R 满足 exp(-R²/2σ²)=0.01 → R=3.0349σ。
# 比值 = 3.0349/1.4142 = 2.146。申报书要求"测量误差限按 99% 覆盖水平校准"，
# 就用这个系数，不要各自拍脑袋乘 2 或乘 3。
RAYLEIGH_99_OVER_RMS = 2.146


def radius_to_99pct(rms_radius: float | NDArray) -> float | NDArray:
    """RMS 误差半径 → 99% 覆盖半径。适用于二维、各向同性、高斯误差。"""
    return RAYLEIGH_99_OVER_RMS * np.asarray(rms_radius)


@dataclass(frozen=True)
class GroupPlan:
    """模块 3（同学 C）的输出：本时隙的分组方案。

    一个"组"= 共用同一条 SRS 序列的用户集合。组内用户同时测量，组间正交。

    fallback 记录降级原因（如 "split_small_gap" / "widen_window" /
    "isolate"），这是消融实验和结题报告里解释"为什么这组没复用"的依据，
    必须真的填，不能全部留空。
    """

    slot: int
    groups: tuple[tuple[int, ...], ...]      # 每组的 uid 元组
    n_symbols: tuple[int, ...]               # 与 groups 一一对应的符号数 L
    fallback: dict[int, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.groups) != len(self.n_symbols):
            raise ValueError("groups 与 n_symbols 长度必须一致")
        seen: set[int] = set()
        for g in self.groups:
            if len(g) == 0:
                raise ValueError("存在空分组")
            if seen & set(g):
                raise ValueError("同一用户被分到多个组")
            seen |= set(g)
        for n in self.n_symbols:
            if n < 1:
                raise ValueError("每组符号数至少为 1")

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def n_users(self) -> int:
        return sum(len(g) for g in self.groups)

    def group_of(self, uid: int) -> int:
        """返回 uid 所属组的索引；不存在的用户抛 KeyError（不要静默返回 -1）。"""
        for i, g in enumerate(self.groups):
            if uid in g:
                return i
        raise KeyError(f"用户 {uid} 不在任何分组中")


@dataclass(frozen=True)
class MeasurementTask:
    """一个已排定的测量任务 = 一次时频资源占用。"""

    slot: int
    group_id: int
    uids: tuple[int, ...]
    n_symbols: int
    window: BeamWindow

    @property
    def resource_units(self) -> int:
        """按申报书口径本次测量消耗的资源单位数。

        关键：等于 n_symbols，**与组内用户数无关**。8 个用户共用一条序列
        测 4 个符号，仍是 4 个单位；正交方案则是 4×8 = 32 个单位。
        这个 8 倍差就是复用的全部收益来源。
        """
        return int(self.n_symbols)

    @property
    def user_measurements(self) -> int:
        """被测量的用户次数（统计量，不是资源量）。与 resource_units 分开记。"""
        return len(self.uids)


@dataclass(frozen=True)
class Schedule:
    """模块 4（同学 D）的输出：一段时间的调度方案。"""

    tasks: tuple[MeasurementTask, ...]
    missed_uids: tuple[int, ...] = ()   # 未能在截止期前安排的（超期）用户

    def __post_init__(self) -> None:
        slots = [t.slot for t in self.tasks]
        if slots != sorted(slots):
            raise ValueError("tasks 必须按 slot 非递减排列，便于事件驱动推进")


@dataclass(frozen=True)
class ErrorBudget:
    """模块 4 的误差预算：决定"每个用户最晚什么时候必须再测一次"。

    申报书条件（式 1）：

        测量误差限 + 固定偏差限 + 预测误差限 ≤ 方向误差容限

    其中三项都以 RMS 误差半径计（半径可加的前提是各项独立且近似各向同性；
    严格来说应做平方和开方，本契约保留线性相加以匹配申报书写法，但在实现
    时建议同时跑 sqrt 版本做敏感性对比——线性相加偏保守，会让方法看起来
    比实际差，反而更安全）。

    固定偏差**不可通过加长测量消除**，这是本项目最值得强调的一点：若
    fixed_bias 已接近 tol_radius，再怎么加符号也没用，这时只能改姿态/星历
    精度或放宽容限。is_feasible() 为 False 时，算法必须直接报告"不可达"，
    而不是给出一个虚假的极小间隔。
    """

    tol_radius: float        # 方向误差容限 ε，RMS 半径
    meas_radius: float       # 测量误差限（已按 99% 覆盖校准）
    fixed_bias: float        # 姿态 + 星历造成的固定偏差
    pred_rate: float         # 预测误差增长率，方向余弦半径 / 秒
    proc_delay_s: float = 0.0  # 处理延迟，占用预算但不可测

    def __post_init__(self) -> None:
        for name in ("tol_radius", "meas_radius", "fixed_bias", "pred_rate"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负")
        if self.proc_delay_s < 0:
            raise ValueError("proc_delay_s 不能为负")

    def is_feasible(self) -> bool:
        """容限是否至少覆盖"测完就已经超了"的那部分。"""
        return self.tol_radius > self.meas_radius + self.fixed_bias

    def max_interval_s(self) -> float:
        """最大更新间隔 T_max，单位秒。

            T_max = (tol - meas - fixed) / pred_rate - proc_delay

        不可行时返回 0.0（表示必须立即/连续测量），而不是负数。
        """
        if not self.is_feasible():
            return 0.0
        if self.pred_rate <= 0.0:
            return math.inf          # 静止用户且无漂移：理论可无限期不更新
        raw = (self.tol_radius - self.meas_radius - self.fixed_bias) / self.pred_rate
        return max(0.0, raw - self.proc_delay_s)

    def slack(self) -> float:
        """剩余预算余量。为负说明该配置下方法失效——这是要报告的结论，不是错误。"""
        return self.tol_radius - self.meas_radius - self.fixed_bias


# ==============================================================================
# 第 5 节  资源计数（考核指标 20% 的分子分母都从这里出）
# ==============================================================================


class MeasureKind(str, Enum):
    """测量的性质。分开统计是因为申报书明确要求把初始化/重捕与控制开销
    单独记录，且初始化阶段的超期率与持续跟踪阶段分开报告。"""

    TRACK = "track"          # 持续跟踪
    ACQUIRE = "acquire"      # 首次方向获取
    REACQUIRE = "reacquire"  # 失锁后重新获取
    CONTROL = "control"      # 信令与控制开销


@dataclass
class ResourceLedger:
    """资源账本。**所有**方案（含基线）都必须用同一个账本记录，否则 20% 无从比较。

    计数口径（与申报书逐字对应）
    ----------------------------
    单位 = 给定带宽内、一个 OFDM 符号上的一条正交 SRS 序列。
    · 共用序列不按用户数重复计数 → 一组的消耗只记 L，不记 L×|组|。
    · 多符号测量按实际符号数累加 → 记 L。
    · 初始获取 / 重新获取 / 控制开销计入总成本，但按 kind 分开存。

    另记两项非资源量，用于解释收益来源：
    · user_measurements：被测量的用户次数（复用下会远大于资源单位数）。
    · time_freq_prb_symbols：PRB·符号，反映真实时频占用。
    """

    sequence_symbols: dict[MeasureKind, int] = field(default_factory=dict)
    user_measurements: dict[MeasureKind, int] = field(default_factory=dict)
    time_freq_prb_symbols: dict[MeasureKind, int] = field(default_factory=dict)

    def record(
        self,
        kind: MeasureKind,
        n_users: int,
        n_symbols: int,
        bandwidth_prb: int,
    ) -> None:
        """登记一次测量。

        参数
        ----
        kind : MeasureKind
        n_users : int
            本次共用同一序列的用户数。**不参与资源计数**，只记用户测量次数。
        n_symbols : int
            本次占用的 OFDM 符号数 L。
        bandwidth_prb : int
            带宽，PRB 数。
        """
        if n_users < 1:
            raise ValueError("n_users 至少为 1")
        if n_symbols < 1:
            raise ValueError("n_symbols 至少为 1")

        self.sequence_symbols[kind] = self.sequence_symbols.get(kind, 0) + int(n_symbols)
        self.user_measurements[kind] = self.user_measurements.get(kind, 0) + int(n_users)
        self.time_freq_prb_symbols[kind] = (
            self.time_freq_prb_symbols.get(kind, 0) + int(n_symbols) * int(bandwidth_prb)
        )

    def record_task(self, task: MeasurementTask, bandwidth_prb: int) -> None:
        """直接从 MeasurementTask 记账，避免手抄参数出错。"""
        self.record(
            kind=MeasureKind.TRACK,
            n_users=task.user_measurements,
            n_symbols=task.n_symbols,
            bandwidth_prb=bandwidth_prb,
        )

    @property
    def total_units(self) -> int:
        """总资源单位数（所有类别求和）。这是"探测资源消耗"这一指标的分子。"""
        return int(sum(self.sequence_symbols.values()))

    @property
    def total_user_measurements(self) -> int:
        return int(sum(self.user_measurements.values()))

    def reuse_gain(self, baseline_units: float) -> float:
        """相对基线的资源节省比例，返回 [0,1)。20% 目标就看这个数。

        参数
        ----
        baseline_units : float
            **同一场景、同一精度要求、调优后**的正交+固定周期方案的资源单位。
            基线没调优就算出来的收益毫无意义，这一点申报书也强调了。
        """
        if baseline_units <= 0:
            raise ValueError("baseline_units 必须为正")
        return 1.0 - self.total_units / float(baseline_units)


# ==============================================================================
# 第 6 节  模块接口（谁实现什么）
# ==============================================================================
#
# 每个 Protocol 只规定"输入什么、输出什么"，不规定怎么算。实现方在自己的
# 文件里写一个普通类，方法签名对上即可，无需继承——Python 的结构化子类型
# 让这成为可能，也让四个人可以不改同一个文件。


@runtime_checkable
class SceneGenerator(Protocol):
    """模块 1 · 同学 A

    职责：给定布局、用户数、种子，生成一个自洽的 Scene（含轨迹、用户、真值）。
    必须保证：同 (layout, n_users, seed) → 完全相同的 Scene（可复现）。
    """

    def __call__(
        self,
        cfg: SystemConfig,
        layout: str,
        n_users: int,
        seed: int,
    ) -> Scene: ...


@runtime_checkable
class Observer(Protocol):
    """模块 1 · 同学 A

    职责：把 Scene + 本时隙要测的用户 + 观测窗 → 波束域观测。
    必须保证：噪声方差严格按第 3 节的 SNR 定义生成；同一 seed 下可复现；
    **不得**把真值写进 Observation。
    """

    def observe(
        self,
        scene: Scene,
        slot: int,
        uids: Sequence[int],
        window: BeamWindow,
        srs: SRSConfig,
        theta_pred: NDArray,
    ) -> Observation: ...


@runtime_checkable
class Estimator(Protocol):
    """模块 2 · 同学 B

    职责：从 Observation 估出方向，并给出同配置下的 CRB。
    crb() 独立于 estimate() 暴露，是为了让模块 3 能在**不产生观测**的情况下
    快速评估候选组能否复用——这是整组 CRB 检查能做到实时的前提。
    """

    def estimate(self, obs: Observation) -> Estimate: ...

    def crb(
        self,
        window: BeamWindow,
        theta_uv: NDArray,
        snr_db: float,
        n_symbols: int,
        cfg: SystemConfig,
    ) -> NDArray:
        """返回 (K,) 的 RMS 误差半径下界。口径见 Estimate 类注释。"""
        ...


@runtime_checkable
class Predictor(Protocol):
    """模块 4 · 同学 D（v1.0.1 新增）

    职责：把上一拍的估计外推为下一拍的预测方向。Observer.observe 与
    Grouper.group 都以 theta_pred / cov_pred 为输入，本协议是它们唯一的
    生产者——没有它，"角度跟踪"的闭环就缺一段胶水。

    为什么归同学 D：预测 = 运动模型 + 上次估计。按用户类型（静止/行人/
    车辆）建运动模型本来就是误差预算 pred_rate 的地基，同一套知识顺手
    产出预测器；且预测器嵌在调度主循环里（测完 → 更新 → 排下一轮），
    与 sched.py 天然同处。同学 A 在平台里把它串进闭环，同学 B 保证
    Estimate 携带可外推的不确定度（cov，或经 crb_radius 各向同性换算）。

    边界：scene 只用于读取用户的标称运动状态（UserState 是可观测状态，
    不违反铁律 2）；不得读取 TruthTable。尚无历史估计的用户（首次获取）
    走 MeasureKind.ACQUIRE 路径，不在本协议范围内。
    """

    def predict(
        self,
        slot_from: int,
        slot_to: int,
        est: Estimate,
        scene: Scene,
    ) -> tuple[NDArray, NDArray | None]:
        """外推预测。

        参数
        ----
        slot_from, slot_to : int
            上次测量的时隙与目标时隙，slot_to - slot_from 即预测跨度。
        est : Estimate
            上一拍的估计结果。cov 为 None 时按 Σ ≈ (crb_radius²/2)·I 近似。
        scene : Scene
            场景，用于读取用户类型与标称运动状态。

        返回
        ----
        (theta_pred, cov_pred)
            theta_pred : (K, 2)，顺序与 est.uids 一致。
            cov_pred : (K, 2, 2) 或 None。None 仅限调用方显式声明不需要
            不确定度的场合；Grouper 与误差预算都必须显式处理 None 分支，
            不要默认返回 None。
        """
        ...


@runtime_checkable
class Grouper(Protocol):
    """模块 3 · 同学 C

    职责：根据预测方向与预测不确定度，决定本时隙哪些用户共用 SRS。
    必须保证：返回的 GroupPlan 里每个用户至多出现一次；降级原因写进 fallback。
    """

    def group(
        self,
        slot: int,
        theta_pred: NDArray,
        cov_pred: NDArray | None,
        uids: Sequence[int],
        cfg: SystemConfig,
        snr_db: float,
    ) -> GroupPlan: ...


@runtime_checkable
class Scheduler(Protocol):
    """模块 4 · 同学 D

    职责：依据各用户截止期、可用资源、当前分组，产出本时段的 Schedule。
    必须保证：未排上的用户进 missed_uids，不能悄悄丢掉。
    """

    def plan(
        self,
        slot: int,
        deadline_slot: dict[int, int],
        groups: GroupPlan,
        ledger: ResourceLedger,
    ) -> Schedule: ...


@runtime_checkable
class Metrics(Protocol):
    """模块 4 · 同学 D

    职责：统一口径的指标计算。所有方案的指标都必须走这里，禁止各自实现一份
    误差公式——两套公式算出的 RMSE 能差 3 dB，而且看不出哪套是对的。
    """

    def update(self, est: Estimate, truth: TruthTable) -> None: ...

    def summary(self) -> dict[str, float]:
        """至少包含：rmse_radius、radius_p99、overdue_rate、max_gap_slot、
        overhead_units、trackable_scale。概率类指标须附样本量。"""
        ...


# 模块归属速查表。新同学接手时先看这张表，再去看具体文件。
#
# 值一律是**代号**（同学 A/B/C/D），不是真实姓名——理由见文件头"模块归属"
# 一节。代码里若要打印负责人，直接查这张表，不要在别处再写一遍名字：
#
#     print(f"Estimator 由 {OWNERS['Estimator']} 负责")
OWNERS: dict[str, str] = {
    "SceneGenerator": "同学 A",
    "Observer": "同学 A",
    "Estimator": "同学 B",
    "Predictor": "同学 D",
    "Grouper": "同学 C",
    "Scheduler": "同学 D",
    "Metrics": "同学 D",
}


# ==============================================================================
# 第 7 节  自检
# ==============================================================================
#
# 这些检查不是"测试算法对不对"，而是"测试四个人的约定是否自洽"。任何一条
# FAIL 都说明契约本身有问题，必须在写算法前修掉。


def _test_config_consistency() -> list[str]:
    """物理参数之间是否自洽。"""
    msgs = []
    cfg = SystemConfig()

    # d/λ 必须落在 (0, 0.5]，否则可见区混叠。构造时已断言，这里复核数值。
    if not (0.44 < cfg.d_over_lambda_ul < 0.47):
        msgs.append(f"d/λ_ul={cfg.d_over_lambda_ul:.4f} 与设计值 0.453 不符")
    if not (0.48 < cfg.d_over_lambda_dl < 0.51):
        msgs.append(f"d/λ_dl={cfg.d_over_lambda_dl:.4f} 与设计值 0.496 不符（应接近半波长）")

    # 服务区半径量级约 850 km。
    r_km = cfg.service_radius_m / 1e3
    if not (800 < r_km < 900):
        msgs.append(f"服务区半径 {r_km:.0f} km 与预期 850 km 不符")

    # 最大离轴角约 52.3°。
    if not (50 < cfg.max_off_nadir_deg < 55):
        msgs.append(f"最大离轴角 {cfg.max_off_nadir_deg:.1f}° 与预期 52.3° 不符")

    return msgs


def _test_no_aliasing_margin() -> list[str]:
    """观测窗是否会越过 DFT 周期边界（越界后波束指向会跳变，是隐蔽 bug）。"""
    msgs = []
    cfg = SystemConfig()
    u_max = math.sin(math.radians(cfg.max_off_nadir_deg))
    k_max = cfg.n_u * cfg.d_over_lambda_ul * u_max
    for hw in (1, 2, 3):
        if k_max + hw > cfg.n_u // 2:
            msgs.append(
                f"半宽 {hw} 时波束索引上界 {k_max + hw:.1f} 超过 N/2={cfg.n_u // 2}，"
                "窗会越界"
            )
    return msgs


def _test_steering_norm() -> list[str]:
    """流形能量 ||a||² 应等于 N_ant，SNR 定义依赖这个性质。"""
    msgs = []
    cfg = SystemConfig()
    a = steering_upa(np.array([0.1, -0.2]), cfg)
    if a.shape != (cfg.n_ant,):
        msgs.append(f"单用户流形形状应为 ({cfg.n_ant},)，收到 {a.shape}")
    nrm = float(np.vdot(a, a).real)
    if not math.isclose(nrm, cfg.n_ant, rel_tol=1e-9):
        msgs.append(f"||a||²={nrm:.3f}，应为 N_ant={cfg.n_ant}")
    return msgs


def _test_beam_peak() -> list[str]:
    """方向 → 波束空间的峰值位置必须落在 rint(N·ψ)。

    这一条守的是"符号约定"。写反的话峰值跑到 -Nψ，角度整体镜像，误差曲线
    看着正常但全错。
    """
    msgs = []
    cfg = SystemConfig()
    for (u, v) in [(0.10, -0.20), (-0.30, 0.15), (0.0, 0.0)]:
        uv = np.array([u, v])
        a = steering_upa(uv, cfg)
        Y = to_beamspace(a, make_window(uv, cfg, 3), cfg)
        win = make_window(uv, cfg, 3)
        idx = int(np.argmax(np.abs(Y)))
        k_hat = (int(win.k_u[idx]), int(win.k_v[idx]))
        k_exp = (
            int(np.rint(cfg.n_u * cfg.d_over_lambda_ul * u)),
            int(np.rint(cfg.n_v * cfg.d_over_lambda_ul * v)),
        )
        if k_hat != k_exp:
            msgs.append(f"方向 ({u},{v}) 的峰值波束 {k_hat}，应为 {k_exp}（符号约定可能反了）")
    return msgs


def _test_beamspace_unitary() -> list[str]:
    """酉变换应保持能量：||Y_beam||² = ||y_elem||²。"""
    msgs = []
    cfg = SystemConfig()
    rng = np.random.default_rng(0)
    y = rng.normal(size=cfg.n_ant) + 1j * rng.normal(size=cfg.n_ant)
    win = make_window(np.array([0.0, 0.0]), cfg, 3)
    Y_win = to_beamspace(y, win, cfg)
    e_win = float(np.sum(np.abs(Y_win) ** 2))

    # 窗只保留部分波束，能量必然少于全量，但必须 ≤ 全量且量级合理。
    e_all = float(np.sum(np.abs(y) ** 2))
    if e_win > e_all * (1 + 1e-9):
        msgs.append(f"窗内能量 {e_win:.3f} 超过全量能量 {e_all:.3f}，归一化错误")

    # 全窗检查：用全波束应严格等能量。
    full = BeamWindow(
        k_u=np.repeat(np.arange(-cfg.n_u // 2, cfg.n_u // 2), cfg.n_v),
        k_v=np.tile(np.arange(-cfg.n_v // 2, cfg.n_v // 2), cfg.n_u),
        half_width_modes=cfg.n_u // 2,
    )
    e_full = float(np.sum(np.abs(to_beamspace(y, full, cfg)) ** 2))
    if not math.isclose(e_full, e_all, rel_tol=1e-9):
        msgs.append(f"全波束能量 {e_full:.6f} ≠ 阵元域 {e_all:.6f}，酉归一化不对")
    return msgs


def _test_snr_definition() -> list[str]:
    """由 SNR 反解出的 σ² 代回后应还原同一 SNR；L 个符号应带来 10log10(L) 增益。"""
    msgs = []
    cfg = SystemConfig()
    for snr_db in (20.0, 30.0, 40.0):
        s2 = noise_var_from_array_snr(snr_db, cfg)
        back = array_snr_from_noise(s2, cfg)
        if not math.isclose(back, snr_db, abs_tol=1e-9):
            msgs.append(f"SNR 往返不一致：设定 {snr_db}，回读 {back:.6f}")

    gain4 = snr_after_symbols(20.0, 4) - 20.0
    if not math.isclose(gain4, 6.0206, abs_tol=1e-3):
        msgs.append(f"4 符号增益 {gain4:.4f} dB，应为 6.0206 dB")
    return msgs


def _test_resource_counting() -> list[str]:
    """复用计数口径：8 用户共测 4 符号 = 4 单位，不是 32。"""
    msgs = []
    cfg = SystemConfig()
    win = make_window(np.array([0.0, 0.0]), cfg, 1)

    task = MeasurementTask(
        slot=0,
        group_id=0,
        uids=tuple(range(8)),
        n_symbols=4,
        window=win,
    )
    if task.resource_units != 4:
        msgs.append(f"8 用户共用测 4 符号应记 4 单位，实际 {task.resource_units}")

    led_reuse = ResourceLedger()
    led_reuse.record_task(task, bandwidth_prb=96)

    led_orth = ResourceLedger()
    for u in range(8):
        led_orth.record(MeasureKind.TRACK, n_users=1, n_symbols=4, bandwidth_prb=96)

    if led_reuse.total_units != 4:
        msgs.append(f"复用账本总量 {led_reuse.total_units}，应为 4")
    if led_orth.total_units != 32:
        msgs.append(f"正交基线总量 {led_orth.total_units}，应为 32")

    gain = led_reuse.reuse_gain(led_orth.total_units)
    if not math.isclose(gain, 0.875, abs_tol=1e-9):
        msgs.append(f"复用收益 {gain:.4f}，应为 0.875（8 用户共测的极端上限）")

    # 用户测量次数不受复用影响，必须分开记。
    if led_reuse.total_user_measurements != 8:
        msgs.append("复用账本丢失了用户测量次数（应仍为 8）")
    return msgs


def _test_error_budget() -> list[str]:
    """误差预算公式与不可行判定。"""
    msgs = []
    b = ErrorBudget(
        tol_radius=1.0e-3,
        meas_radius=2.0e-4,
        fixed_bias=3.0e-4,
        pred_rate=1.0e-6,
        proc_delay_s=10.0,
    )
    t_max = b.max_interval_s()
    expect = (1.0e-3 - 2.0e-4 - 3.0e-4) / 1.0e-6 - 10.0
    if not math.isclose(t_max, expect, rel_tol=1e-9):
        msgs.append(f"T_max={t_max:.1f}s，应为 {expect:.1f}s")

    # 固定偏差吃满预算 → 不可行，且必须返回 0 而不是负数。
    bad = ErrorBudget(tol_radius=1.0e-3, meas_radius=2.0e-4,
                      fixed_bias=8.5e-4, pred_rate=1.0e-6)
    if bad.is_feasible():
        msgs.append("固定偏差已占 85% 预算，应判为不可行")
    if bad.max_interval_s() != 0.0:
        msgs.append("不可行时 max_interval_s 应返回 0.0")

    # 静止用户且无漂移 → 无需更新。
    static_user = ErrorBudget(tol_radius=1e-3, meas_radius=1e-4,
                              fixed_bias=1e-4, pred_rate=0.0)
    if static_user.max_interval_s() != math.inf:
        msgs.append("pred_rate=0 时应返回 inf")
    return msgs


def _test_rayleigh_coefficient() -> list[str]:
    """99% 覆盖系数用蒙特卡洛复核，别信手算。"""
    msgs = []
    rng = np.random.default_rng(12345)
    n = 400_000
    e = rng.normal(size=(n, 2))
    r = np.hypot(e[:, 0], e[:, 1])
    ratio = float(np.quantile(r, 0.99) / np.sqrt(np.mean(r ** 2)))
    if not math.isclose(ratio, RAYLEIGH_99_OVER_RMS, rel_tol=2e-3):
        msgs.append(
            f"99%/RMS 实测 {ratio:.4f}，契约常数 {RAYLEIGH_99_OVER_RMS}，偏差过大"
        )
    return msgs


def _test_shape_guards() -> list[str]:
    """形状守卫是否真的会拦住退化输入（单用户写成 (2,) 这类）。"""
    msgs = []
    cfg = SystemConfig()
    win = make_window(np.array([0.0, 0.0]), cfg, 1)
    try:
        Observation(
            slot=0,
            uids=(0,),
            window=win,
            Y=np.zeros((win.n_beams, 4), dtype=complex),
            srs=SRSConfig(n_symbols=4),
            theta_pred=np.zeros(2),          # 故意写成 (2,)
        )
    except ValueError:
        pass
    else:
        msgs.append("Observation 未拦住 theta_pred=(2,) 的退化形状")

    try:
        Estimate(
            slot=0,
            uids=(0,),
            theta_hat=np.zeros((1, 2)),
            crb_radius=np.array([1e-3]),
            status=EstimateStatus.OK,
            cov=np.zeros((2, 2)),            # 故意缺 K 维
        )
    except ValueError:
        pass
    else:
        msgs.append("Estimate 未拦住 cov=(2,2) 的退化形状（应为 (K,2,2) 或 None）")
    return msgs


# 自检登记表。新增字段时请顺手加一条检查，尤其是形状与口径类的。
_SELF_TESTS = [
    ("物理参数自洽", _test_config_consistency),
    ("观测窗不越界", _test_no_aliasing_margin),
    ("流形能量 ||a||²=N_ant", _test_steering_norm),
    ("波束峰值位置（符号约定）", _test_beam_peak),
    ("波束域酉归一化", _test_beamspace_unitary),
    ("SNR 定义往返", _test_snr_definition),
    ("资源计数口径", _test_resource_counting),
    ("误差预算公式", _test_error_budget),
    ("99% 覆盖系数", _test_rayleigh_coefficient),
    ("形状守卫", _test_shape_guards),
]


def self_test(verbose: bool = True) -> bool:
    """跑全部契约自检，返回是否全部通过。"""
    ok = True
    for name, fn in _SELF_TESTS:
        try:
            msgs = fn()
        except Exception as exc:                      # noqa: BLE001
            ok = False
            if verbose:
                print(f"  [ERROR] {name}: {type(exc).__name__}: {exc}")
            continue

        if msgs:
            ok = False
            if verbose:
                print(f"  [FAIL ] {name}")
                for m in msgs:
                    print(f"          · {m}")
        elif verbose:
            print(f"  [PASS ] {name}")

    if verbose:
        print()
        print(f"契约版本 {CONTRACT_VERSION} · " + ("全部通过" if ok else "存在失败项，先修契约再写算法"))
    return ok


if __name__ == "__main__":
    print("=" * 72)
    print("srs_track.contract 自检")
    print("=" * 72)

    cfg = SystemConfig()
    print(f"  阵元数 N_ant      : {cfg.n_ant}  ({cfg.n_u}×{cfg.n_v})")
    print(f"  d/λ 上行 / 下行   : {cfg.d_over_lambda_ul:.4f} / {cfg.d_over_lambda_dl:.4f}")
    print(f"  服务区半径        : {cfg.service_radius_m / 1e3:.0f} km")
    print(f"  最大离轴角        : {cfg.max_off_nadir_deg:.1f}°")
    print(f"  20 dB 时 σ²       : {noise_var_from_array_snr(20.0, cfg):.3e}")
    print()

    success = self_test(verbose=True)
    raise SystemExit(0 if success else 1)
