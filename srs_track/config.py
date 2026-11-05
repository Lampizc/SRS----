# -*- coding: utf-8 -*-
"""srs_track.config — config.yaml 进入代码的唯一通道（参数唯一真源的"加载校验"一环）。

规则（见项目根 config.yaml 文件头）
--------------------------------
· 全项目系统参数只允许从 config.yaml 进入，任何模块禁止硬编码；
· 本文件做三层校验后构造 frozen SystemConfig / SRSConfig：
    1) 契约版本核对：meta.contract_version_expected 必须等于 contract.CONTRACT_VERSION，
       不匹配说明配置与契约脱节，拒绝加载；
    2) 键名白名单：任何一节出现未知键直接报错——拼错参数名不允许静默通过；
    3) 取值校验：类型/范围检查（含 PyYAML 科学计数法陷阱的专门报错），
       最终由 SystemConfig / SRSConfig 自身的 __post_init__ 兜底。
· system: 节缺键 = 采用契约默认值。默认值不复制到 yaml，避免两处维护。

用法
----
    from srs_track.config import load_config
    pc  = load_config("config.yaml")     # 通常从项目根运行
    cfg = pc.system                      # frozen SystemConfig
    srs = pc.srs_default                 # SRSConfig 默认值
    exp = pc.experiment                  # 实验矩阵原样透传（dict）
    seeds(pc)                            # → [base_seed, ..., base_seed+n_seeds-1]

自检（校验器内部自洽，任何时候都应 PASS）：
    python -m srs_track.config          # 从项目根运行
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, fields

from .contract import CONTRACT_VERSION, SRSConfig, SystemConfig, UserType

try:
    import yaml
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "加载 config.yaml 需要 PyYAML（本机 3.14 环境已装 6.0.3；"
        "若换了环境请先 pip install pyyaml）"
    ) from _exc


# ==============================================================================
# 校验工具
# ==============================================================================

_SCI_HINT = (
    "PyYAML 只把'带小数点且指数带符号'的写法解析为浮点数："
    "请写 1.995e+9、5.0e-4，而不是 1.995e9 / 0.5e-3（后者会被当成字符串）。"
)


def _require_number(section: str, key: str, value, *, integral: bool = False) -> None:
    """数值字段类型检查。integral=True 时要求整数。bool 不是数值。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"config[{section}].{key} 必须是{'整数' if integral else '数值'}，"
            f"收到 {type(value).__name__}（值={value!r}）。{_SCI_HINT}"
        )
    if integral and not isinstance(value, int):
        raise TypeError(
            f"config[{section}].{key} 必须是整数，收到浮点数 {value!r}"
            "（阵列规模/符号数/种子这类离散量不允许写 64.0）"
        )
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"config[{section}].{key} 不能是 inf/nan")


def _require_bool(section: str, key: str, value) -> None:
    if not isinstance(value, bool):
        raise TypeError(
            f"config[{section}].{key} 必须是布尔值，收到 {type(value).__name__}"
            f"（值={value!r}）。YAML 里写 true/false，不要加引号。"
        )


def _check_keys(section: str, mapping: dict, allowed: tuple[str, ...]) -> None:
    unknown = set(mapping) - set(allowed)
    if unknown:
        raise KeyError(
            f"config[{section}] 存在未知键 {sorted(unknown)}；"
            f"合法键：{sorted(allowed)}。新增键须先在 srs_track/config.py "
            "登记白名单并通过组会确认。"
        )


def _require_number_list(section: str, key: str, value, *, integral: bool = False) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(f"config[{section}].{key} 必须是非空列表")
    for i, v in enumerate(value):
        _require_number(f"{section}.{key}[{i}]", "item", v, integral=integral)
    if integral and any(v < 1 for v in value):
        raise ValueError(f"config[{section}].{key} 的元素必须 ≥ 1")


# ==============================================================================
# 白名单登记表（新增配置键：先改 yaml，再在这里登记，自检里补一条校验）
# ==============================================================================

_TOP_SECTIONS = ("meta", "system", "srs_default", "experiment")
_META_KEYS = ("schema_version", "contract_version_expected")
_SRS_KEYS = tuple(f.name for f in fields(SRSConfig))
_SYSTEM_KEYS = tuple(f.name for f in fields(SystemConfig))

_EXP_KEYS = (
    "layouts",
    "user_types",
    "snr_db_list",
    "window_half_width_modes",
    "group_sizes",
    "n_symbols_list",
    "n_users_list",
    "n_seeds",
    "base_seed",
    "error_budget_defaults",
    "layout_params",
    "demo",
)

_BUDGET_KEYS = (
    "tol_radius_rms",     # 方向误差容限 ε（RMS 半径）→ ErrorBudget.tol_radius
    "meas_radius_rms",    # 测量误差限（99% 覆盖校准后）→ ErrorBudget.meas_radius
    "fixed_bias",         # 姿态+星历固定偏差 → ErrorBudget.fixed_bias
    "pred_rate",          # 预测误差增长率 → ErrorBudget.pred_rate
    "proc_delay_s",       # 处理延迟 → ErrorBudget.proc_delay_s
)

_DEMO_KEYS = ("n_users", "n_slots", "snr_db", "window_half_width_modes", "n_symbols")

_LAYOUTS = ("uniform", "hotspot", "corridor")
_USER_TYPES = tuple(t.value for t in UserType)


# ==============================================================================
# 加载入口
# ==============================================================================


@dataclass(frozen=True)
class ProjectConfig:
    """load_config 的返回值：契约对象 + 实验矩阵。"""

    system: SystemConfig      # frozen，全模块共享同一实例
    srs_default: SRSConfig    # 默认 SRS 资源配置，单次测量可覆盖
    experiment: dict          # 实验矩阵原样透传（校验过结构，不改值）
    path: str                 # 来源文件路径，便于日志回溯


def load_config(path: str | pathlib.Path = "config.yaml") -> ProjectConfig:
    """读取并校验 yaml，构造 frozen 的契约对象。

    参数
    ----
    path : str | Path
        配置文件路径。默认项目根的 config.yaml；调试小阵列传 config_debug.yaml。

    返回
    ----
    ProjectConfig

    异常
    ----
    任何键名 / 类型 / 版本 / 取值问题都以异常形式报出，绝不静默降级——
    配置错误宁可当场崩，也不要让四个人在错误参数上各跑一天实验。
    """
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在：{p.resolve()}（请从项目根运行）")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: 顶层必须是映射（键值对），收到 {type(raw).__name__}")
    _check_keys("<顶层>", raw, _TOP_SECTIONS)

    # --- 1) 契约版本核对 -------------------------------------------------------
    meta = raw.get("meta") or {}
    _check_keys("meta", meta, _META_KEYS)
    expected = meta.get("contract_version_expected")
    if expected is not None and expected != CONTRACT_VERSION:
        raise ValueError(
            f"{p}: 契约版本不符——config 期望 {expected!r}，当前契约 {CONTRACT_VERSION!r}。"
            "先确认契约改动是否影响本文件（看 contract.py 的版本注释），"
            "再同步 meta.contract_version_expected。"
        )

    # --- 2) system → SystemConfig（缺键 = 契约默认值）--------------------------
    system_raw = raw.get("system") or {}
    _check_keys("system", system_raw, _SYSTEM_KEYS)
    for key in ("n_u", "n_v"):
        if key in system_raw:
            _require_number("system", key, system_raw[key], integral=True)
    for key in ("orbit_height_km", "min_elev_deg", "spacing_m",
                "f_ul_hz", "f_dl_hz", "slot_duration_s"):
        if key in system_raw:
            _require_number("system", key, system_raw[key])
    system = SystemConfig(**system_raw)

    # --- 3) srs_default → SRSConfig -------------------------------------------
    srs_raw = raw.get("srs_default") or {}
    _check_keys("srs_default", srs_raw, _SRS_KEYS)
    if "n_symbols" in srs_raw:
        _require_number("srs_default", "n_symbols", srs_raw["n_symbols"], integral=True)
    if "bandwidth_prb" in srs_raw:
        _require_number("srs_default", "bandwidth_prb", srs_raw["bandwidth_prb"], integral=True)
    if "cyclic_shift" in srs_raw:
        _require_number("srs_default", "cyclic_shift", srs_raw["cyclic_shift"], integral=True)
    if "n_ports" in srs_raw:
        _require_number("srs_default", "n_ports", srs_raw["n_ports"], integral=True)
    if "hopping" in srs_raw:
        _require_bool("srs_default", "hopping", srs_raw["hopping"])
    srs = SRSConfig(**srs_raw)

    # --- 4) experiment：白名单 + 结构校验（值原样透传，不加工）------------------
    exp = raw.get("experiment") or {}
    _check_keys("experiment", exp, _EXP_KEYS)

    if "layouts" in exp:
        v = exp["layouts"]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
            raise ValueError("config[experiment].layouts 必须是非空字符串列表")
        bad = set(v) - set(_LAYOUTS)
        if bad:
            raise ValueError(f"layouts 含未知布局 {sorted(bad)}；合法值：{_LAYOUTS}")
    if "user_types" in exp:
        v = exp["user_types"]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v):
            raise ValueError("config[experiment].user_types 必须是非空字符串列表")
        bad = set(v) - set(_USER_TYPES)
        if bad:
            raise ValueError(f"user_types 含未知类型 {sorted(bad)}；合法值：{_USER_TYPES}")
    for key in ("snr_db_list",):
        if key in exp:
            _require_number_list("experiment", key, exp[key])
    for key in ("window_half_width_modes", "group_sizes", "n_symbols_list", "n_users_list"):
        if key in exp:
            _require_number_list("experiment", key, exp[key], integral=True)
    if "n_seeds" in exp:
        _require_number("experiment", "n_seeds", exp["n_seeds"], integral=True)
        if exp["n_seeds"] < 1:
            raise ValueError("config[experiment].n_seeds 必须 ≥ 1（考核要求 ≥5，别往下改）")
    if "base_seed" in exp:
        _require_number("experiment", "base_seed", exp["base_seed"], integral=True)
    if "error_budget_defaults" in exp:
        bud = exp["error_budget_defaults"]
        if not isinstance(bud, dict):
            raise ValueError("config[experiment].error_budget_defaults 必须是映射")
        _check_keys("experiment.error_budget_defaults", bud, _BUDGET_KEYS)
        for k, v in bud.items():
            if v is None:            # null = TODO 未定标，允许，消费方自行兜底
                continue
            _require_number("experiment.error_budget_defaults", k, v)
            if v < 0:
                raise ValueError(f"error_budget_defaults.{k} 不能为负（ErrorBudget 口径）")
    if "layout_params" in exp:
        lp = exp["layout_params"]
        if not isinstance(lp, dict) or not all(isinstance(v, dict) for v in lp.values()):
            raise ValueError("config[experiment].layout_params 必须是'布局名 → 映射'的两层结构")
    if "demo" in exp:
        demo = exp["demo"]
        if not isinstance(demo, dict):
            raise ValueError("config[experiment].demo 必须是映射")
        _check_keys("experiment.demo", demo, _DEMO_KEYS)
        for k, v in demo.items():
            if k == "snr_db":
                _require_number("experiment.demo", k, v)
            else:
                _require_number("experiment.demo", k, v, integral=True)
                if v < 1:
                    raise ValueError(f"demo.{k} 必须 ≥ 1")

    return ProjectConfig(
        system=system,
        srs_default=srs,
        experiment=exp,
        path=str(p),
    )


def seeds(pc: ProjectConfig) -> list[int]:
    """由 base_seed + n_seeds 派生全部实验种子（随机性唯一入口的清单形式）。"""
    exp = pc.experiment
    base = int(exp.get("base_seed", 0))
    n = int(exp.get("n_seeds", 5))
    return [base + i for i in range(n)]


# ==============================================================================
# 自检
# ==============================================================================


def self_test(path: str | pathlib.Path = "config.yaml", verbose: bool = True) -> bool:
    """加载 config.yaml 并核对：能加载、能构造契约对象、与契约默认值对拍。"""
    msgs: list[str] = []
    pc = load_config(path)

    # 往返：system 节写出的每个值必须原样进入 SystemConfig。
    with pathlib.Path(pc.path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    for key, val in (raw.get("system") or {}).items():
        got = getattr(pc.system, key)
        if got != val or type(got) is not type(val):
            msgs.append(f"system.{key} 往返不一致：yaml={val!r} vs 契约对象={got!r}")

    # 与契约默认值对拍：当前 system 节 == 契约默认，若有差异要显式列出。
    default = SystemConfig()
    diffs = [
        f"{f.name}: config={getattr(pc.system, f.name)!r} vs 契约默认={getattr(default, f.name)!r}"
        for f in fields(SystemConfig)
        if getattr(pc.system, f.name) != getattr(default, f.name)
    ]
    if verbose:
        for d in diffs:
            print(f"  [INFO] 与契约默认不同 → {d}")

    if verbose:
        print(f"  [PASS ] 版本核对（契约 {CONTRACT_VERSION}）" if not msgs
              else "  [FAIL ] 往返校验")
        for m in msgs:
            print(f"          · {m}")
        if not msgs:
            print("  [PASS ] yaml → 契约对象往返一致")
        print(f"  [PASS ] 加载成功：{pc.path}")
        print()
        print(f"  n_ant={pc.system.n_ant}  d/λ_ul={pc.system.d_over_lambda_ul:.4f}"
              f"  服务区半径={pc.system.service_radius_m/1e3:.0f} km"
              f"  最大离轴角={pc.system.max_off_nadir_deg:.1f}°")
        print(f"  种子清单 = {seeds(pc)}")
        print(f"  实验矩阵键 = {sorted(pc.experiment)}")
    return not msgs


if __name__ == "__main__":
    print("=" * 72)
    print("srs_track.config 自检（config.yaml → 契约对象）")
    print("=" * 72)
    ok = self_test()
    print("配置校验 " + ("全部通过" if ok else "存在失败项"))
    raise SystemExit(0 if ok else 1)
