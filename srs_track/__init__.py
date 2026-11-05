# -*- coding: utf-8 -*-
"""srs_track — 面向低轨卫星通信的 SRS 角度跟踪与资源调度研究。

模块与负责人
--------------------
    scene.py    场景生成       同学 A（队长）
    array.py    阵列与波束域   同学 A
    obs.py      观测生成       同学 A
    est.py      角度估计       同学 B
    crb.py      Fisher/CRB     同学 B
    group.py    复用分组       同学 C
    budget.py   误差预算       同学 D
    sched.py    EDF 调度       同学 D
    metrics.py  指标统计       同学 D

所有模块之间只通过 contract.py 定义的类型通信。动手前先读 contract.py 文件头。

用法：
    from srs_track.contract import SystemConfig, Scene, Observation, Estimate
    from srs_track import SystemConfig          # 等价，走下面的惰性导出

这里用 PEP 562 的模块级 __getattr__ 做惰性导出，而不是在文件顶部 import contract。
原因是：若在此处立即导入子模块，`python -m srs_track.contract`
会触发 runpy 的 "module found in sys.modules" 警告，输出里多一行噪音，新人看到会以为出错了。
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:                       # 仅供类型检查器与 IDE 补全使用
    from .contract import (  # noqa: F401
        CONTRACT_VERSION,
        BeamWindow,
        ErrorBudget,
        Estimate,
        EstimateStatus,
        GroupPlan,
        MeasureKind,
        MeasurementTask,
        Observation,
        ResourceLedger,
        SRSConfig,
        Schedule,
        Scene,
        SatelliteTrack,
        SystemConfig,
        TruthTable,
        UserState,
        UserType,
    )

# 与 contract.py 的公开名字保持一致。
_EXPORTS = frozenset({
    "CONTRACT_VERSION",
    "BeamWindow",
    "ErrorBudget",
    "Estimate",
    "EstimateStatus",
    "GroupPlan",
    "MeasureKind",
    "MeasurementTask",
    "Observation",
    "ResourceLedger",
    "SRSConfig",
    "Schedule",
    "Scene",
    "SatelliteTrack",
    "SystemConfig",
    "TruthTable",
    "UserState",
    "UserType",
})

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    """按需导入 contract 中的公开名字。"""
    if name in _EXPORTS:
        from . import contract

        return getattr(contract, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
