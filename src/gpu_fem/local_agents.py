"""Local router/configurator/evaluator helpers for the standalone GPU-FEM repo."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Optional

import numpy as np

from .problem_spec import EdgeSupport, PointLoad, ProblemSpec


class SurrogateRouterAgent:
    name = "rule_router"

    def __init__(
        self,
        fem_warmup: int = 10,
        fem_interval: int = 5,
        uncertainty_tol: float = 0.40,
        check_uncertainty: bool = True,
    ) -> None:
        self.fem_warmup = fem_warmup
        self.fem_interval = fem_interval
        self.uncertainty_tol = uncertainty_tol
        self.check_uncertainty = check_uncertainty
        self._prev_beta = 1.0
        self._post_beta_cooldown = 0

    def __call__(self, iteration: int, uncertainty: float, surrogate_ready: bool, beta: float = 1.0, **kwargs) -> bool:
        if not surrogate_ready or iteration <= self.fem_warmup:
            return True
        if (iteration % self.fem_interval) == 0:
            return True
        if abs(beta - self._prev_beta) > 0.5:
            self._prev_beta = beta
            self._post_beta_cooldown = 2
            return True
        self._prev_beta = beta
        if self._post_beta_cooldown > 0:
            self._post_beta_cooldown -= 1
            return True
        if self.check_uncertainty and uncertainty > self.uncertainty_tol:
            return True
        return False


class AggressiveSurrogateRouter(SurrogateRouterAgent):
    name = "aggressive_router"

    def __init__(self) -> None:
        super().__init__(fem_warmup=8, fem_interval=8, check_uncertainty=False)


class ConservativeSurrogateRouter(SurrogateRouterAgent):
    name = "conservative_router"

    def __init__(self) -> None:
        super().__init__(fem_warmup=15, fem_interval=3, uncertainty_tol=0.10, check_uncertainty=True)


class PureFEMRouter:
    name = "pure_fem"

    def __call__(self, iteration: int, uncertainty: float, surrogate_ready: bool, beta: float = 1.0, **kwargs) -> bool:
        return True


class ActiveLearningRouter:
    name = "active_router"

    def __init__(
        self,
        fem_warmup: int = 8,
        delta_rho_tol: float = 0.015,
        uncertainty_tol: float = 0.30,
        compliance_rise_tol: float = 0.05,
    ) -> None:
        self.fem_warmup = fem_warmup
        self.delta_rho_tol = delta_rho_tol
        self.uncertainty_tol = uncertainty_tol
        self.compliance_rise_tol = compliance_rise_tol
        self._prev_rho: Optional[np.ndarray] = None
        self._compliance_history: list[float] = []

    def __call__(
        self,
        iteration: int,
        uncertainty: float,
        surrogate_ready: bool,
        beta: float = 1.0,
        rho: Optional[np.ndarray] = None,
        compliance: Optional[float] = None,
        **kwargs,
    ) -> bool:
        if not surrogate_ready or iteration <= self.fem_warmup:
            return True

        delta_rho = 0.0
        if rho is not None and self._prev_rho is not None:
            delta_rho = float(np.mean(np.abs(rho - self._prev_rho)))
        if rho is not None:
            self._prev_rho = rho.copy()

        compliance_rising = False
        if compliance is not None:
            self._compliance_history.append(compliance)
            if len(self._compliance_history) >= 4:
                recent = self._compliance_history[-4:]
                slope = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10)
                compliance_rising = slope > self.compliance_rise_tol

        return (
            delta_rho > self.delta_rho_tol
            or uncertainty > self.uncertainty_tol
            or compliance_rising
        )


class TemporalSuperResolutionRouter(ActiveLearningRouter):
    """
    Conservative self-healing router for temporal super-resolution surrogate runs.

    This router re-anchors more frequently than the default active-learning
    policy so the surrogate stays close to the latest exact FEM state.
    """

    name = "temporal_sr_router"

    def __init__(
        self,
        fem_warmup: int = 5,
        fem_interval: int = 5,
        delta_rho_tol: float = 0.040,
        uncertainty_tol: float = 0.65,
        compliance_rise_tol: float = 0.03,
    ) -> None:
        super().__init__(
            fem_warmup=fem_warmup,
            delta_rho_tol=delta_rho_tol,
            uncertainty_tol=uncertainty_tol,
            compliance_rise_tol=compliance_rise_tol,
        )
        self.fem_interval = fem_interval

    def __call__(
        self,
        iteration: int,
        uncertainty: float,
        surrogate_ready: bool,
        beta: float = 1.0,
        rho: Optional[np.ndarray] = None,
        compliance: Optional[float] = None,
        **kwargs,
    ) -> bool:
        if not surrogate_ready or iteration <= self.fem_warmup:
            return True
        if (iteration % self.fem_interval) == 0:
            return True
        return super().__call__(
            iteration=iteration,
            uncertainty=uncertainty,
            surrogate_ready=surrogate_ready,
            beta=beta,
            rho=rho,
            compliance=compliance,
            **kwargs,
        )


class LLMRouterAgent(SurrogateRouterAgent):
    name = "llm_router"

    def __init__(self, query_interval: int = 10, fem_warmup: int = 10, uncertainty_tol: float = 0.15) -> None:
        super().__init__(fem_warmup=fem_warmup, fem_interval=query_interval, uncertainty_tol=uncertainty_tol)


class TO3DConfiguratorAgent:
    def configure(self, prompt: str, verbose: bool = False):
        return _fallback_spec(prompt), False, ["Local fallback configurator used."]


def _fallback_spec(prompt: str) -> ProblemSpec:
    prompt_lower = prompt.lower()
    volfrac = 0.3
    if "15%" in prompt_lower or "lightweight" in prompt_lower:
        volfrac = 0.15
    elif "50%" in prompt_lower:
        volfrac = 0.5
    return ProblemSpec(
        Lx=2.0,
        Ly=1.0,
        Lz=0.5,
        nelx=24,
        nely=12,
        nelz=6,
        volfrac=volfrac,
        supports=[EdgeSupport(edge="left", constraint="fixed")],
        loads=[PointLoad(x=2.0, y=0.5, z=0.25, fy=-1.0)],
    )


class TO3DEvaluatorAgent:
    def evaluate(self, result: dict, spec, max_iter: int, use_llm: bool = True, verbose: bool = False):
        return _simple_eval(result)


def _simple_eval(result: dict):
    @dataclass
    class SimpleCheck:
        name: str
        passed: bool
        value: float = 0.0
        threshold: float = 0.0
        message: str = ""

    @dataclass
    class SimpleEvalResult:
        passed: bool
        summary: str
        checks: list = dc_field(default_factory=list)
        rerun_hint: dict = dc_field(default_factory=dict)
        llm_assessment: str = ""

    grayness = float(result.get("final_grayness", 1.0))
    passed = grayness < 0.25 and bool(result.get("best_is_valid", False))
    return SimpleEvalResult(
        passed=passed,
        summary="PASS" if passed else "FAIL - grayness too high or no valid solution.",
        checks=[SimpleCheck("grayness", grayness < 0.25, grayness, 0.25)],
    )
