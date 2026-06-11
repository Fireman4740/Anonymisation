"""
LangGraph Pipeline Graph with resource lifecycle management.

Graph topology (RUPTA adversarial loop):

    detection → llm_detection → llm_verification → anonymization → llm_audit
                                                     ↓  _rupta_router()
                                          ┌── privacy_score > threshold AND iter < max
                                          │              ↓
                                          │       llm_paraphrase
                                          │              │
                                          │              └──→ llm_audit  (loop)
                                          └── else → END

All LLM nodes are feature-flag gated (config.json → features.*).
If LLM is unreachable they degrade gracefully and return empty updates.
"""

from __future__ import annotations

import json
import logging
import atexit
import os
from typing import Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from src.state import PipelineState
from src.nodes.detection.detection_node import DetectionNode
from src.nodes.anonymisation.anonymization_node import AnonymizationNode
from src.nodes.llm.llm_review_node import LLMReviewNode
from src.nodes.llm.llm_verification_node import LLMVerificationNode
from src.nodes.llm.llm_audit_node import LLMAuditNode
from src.nodes.llm.llm_paraphrase_node import LLMParaphraseNode

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

logger = logging.getLogger("PipeGraph")


# ---------------------------------------------------------------------------
# RUPTA conditional router
# ---------------------------------------------------------------------------

_ROUTER_CONFIG_CACHE: dict = {}


def _load_router_config() -> dict:
    """Read config.json once per mtime (the router runs on every audit loop)."""
    config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config.json")
    )
    mtime = os.path.getmtime(config_path)
    cached = _ROUTER_CONFIG_CACHE.get(config_path)
    if cached and cached[0] == mtime:
        return cached[1]
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    _ROUTER_CONFIG_CACHE[config_path] = (mtime, cfg)
    return cfg


def _rupta_router(state: PipelineState) -> str:
    """
    Called after llm_audit.
    Returns "paraphrase" to trigger the RUPTA rewrite loop, or "end" to finish.
    """
    try:
        cfg = _load_router_config()
        rupta_cfg = cfg.get("rupta", {})
        features = cfg.get("features", {})
    except Exception as e:
        logger.warning(f"RUPTA router: could not read config: {e}")
        return "end"

    # Hard off-switches (config.json)
    if not rupta_cfg.get("enabled", False):
        return "end"
    if not (features.get("llm_audit", True) and features.get("llm_paraphrase", True)):
        return "end"

    # --- Runtime overrides depuis state config (Streamlit / benchmark) ---
    runtime = state.get("config", {})
    if runtime.get("rupta_enabled") is False:
        return "end"
    if runtime.get("llm_audit") is False or runtime.get("llm_paraphrase") is False:
        return "end"

    privacy_score: int = state.get("privacy_score", 0)
    iteration: int = state.get("iteration", 0)
    # Priorité aux valeurs du state config, fallback sur config.json
    max_iterations: int = int(
        runtime.get("rupta_max_iterations") or rupta_cfg.get("max_iterations", 3)
    )
    p_threshold: int = int(
        runtime.get("rupta_p_threshold") if runtime.get("rupta_p_threshold") is not None
        else rupta_cfg.get("p_threshold", 15)
    )

    if privacy_score > p_threshold and iteration < max_iterations:
        logger.info(
            f"RUPTA: score={privacy_score} > threshold={p_threshold}, "
            f"iter={iteration}/{max_iterations} → paraphrase"
        )
        return "paraphrase"

    logger.info(
        f"RUPTA: score={privacy_score}, iter={iteration}/{max_iterations} → end"
    )
    return "end"


class GraphResources:
    """
    Manages shared resources for the pipeline graph lifecycle.

    Use as a context manager for automatic cleanup:

        with GraphResources() as resources:
            app = resources.graph
            result = app.invoke({"text": "..."})

    Or manually:

        resources = GraphResources()
        app = resources.graph
        # ... use app ...
        resources.shutdown()
    """

    _instance: Optional["GraphResources"] = None

    def __init__(self, auto_cleanup_on_exit: bool = True):
        """
        Initialize the graph resources.

        Args:
            auto_cleanup_on_exit: If True, register atexit handler for cleanup
        """
        self._detection_node: Optional[DetectionNode] = None
        self._anonymization_node: Optional[AnonymizationNode] = None
        self._llm_review_node: Optional[LLMReviewNode] = None
        self._llm_audit_node: Optional[LLMAuditNode] = None
        self._llm_paraphrase_node: Optional[LLMParaphraseNode] = None
        self._llm_verification_node: Optional[LLMVerificationNode] = None
        self._graph: Optional["CompiledGraph"] = None
        self._shutdown_registered = False

        if auto_cleanup_on_exit:
            self._register_atexit()

    def _register_atexit(self) -> None:
        """Register cleanup on interpreter exit."""
        if not self._shutdown_registered:
            atexit.register(self.shutdown)
            self._shutdown_registered = True

    @property
    def graph(self) -> "CompiledGraph":
        """Get or create the compiled graph."""
        if self._graph is None:
            self._graph = self._create_graph()
        return self._graph

    def _create_graph(self) -> "CompiledGraph":
        """Create and compile the pipeline graph."""
        # 1. Initialize StateGraph
        workflow = StateGraph(PipelineState)

        # 2. Create nodes
        self._detection_node = DetectionNode()
        self._anonymization_node = AnonymizationNode()
        
        # Additive review node replaces old LLMDetection
        self._llm_review_node = LLMReviewNode()
        
        self._llm_verification_node = LLMVerificationNode() # currently optional
        self._llm_audit_node = LLMAuditNode()
        self._llm_paraphrase_node = LLMParaphraseNode()

        # 3. Add nodes to graph
        workflow.add_node("detection", self._detection_node)
        workflow.add_node("anonymization_pass_1", self._anonymization_node)
        workflow.add_node("llm_review", self._llm_review_node)
        workflow.add_node("llm_verification", self._llm_verification_node)
        workflow.add_node("anonymization_pass_2", self._anonymization_node)
        workflow.add_node("llm_audit", self._llm_audit_node)
        workflow.add_node("llm_paraphrase", self._llm_paraphrase_node)

        # 4. Define edges for Additive Anonymization
        workflow.set_entry_point("detection")
        workflow.add_edge("detection", "anonymization_pass_1")
        workflow.add_edge("anonymization_pass_1", "llm_review")
        workflow.add_edge("llm_review", "llm_verification")
        workflow.add_edge("llm_verification", "anonymization_pass_2")
        workflow.add_edge("anonymization_pass_2", "llm_audit")

        # 5. Conditional RUPTA loop after audit
        workflow.add_conditional_edges(
            "llm_audit",
            _rupta_router,
            {"paraphrase": "llm_paraphrase", "end": END},
        )
        workflow.add_edge("llm_paraphrase", "llm_audit")

        # 6. Compile
        logger.info("Pipeline graph compiled (Additive architecture)")
        return workflow.compile()

    def shutdown(self, wait: bool = True) -> None:
        """
        Clean up all resources.

        Args:
            wait: If True, wait for executor tasks to complete
        """
        logger.info("Shutting down pipeline resources...")

        # Clear model caches (free GPU memory)
        try:
            from src.utils.model_cache import clear_all_caches

            clear_all_caches()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Error clearing caches: {e}")

        # Reset instance state
        self._detection_node = None
        self._anonymization_node = None
        self._llm_review_node = None
        self._llm_audit_node = None
        self._llm_verification_node = None
        self._llm_paraphrase_node = None
        self._graph = None

        logger.info("Pipeline resources shutdown complete")

    def __enter__(self) -> "GraphResources":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with cleanup."""
        self.shutdown()
        return None  # Don't suppress exceptions

    @classmethod
    def get_instance(cls) -> "GraphResources":
        """Get singleton instance (for simple use cases)."""
        if cls._instance is None:
            cls._instance = GraphResources(auto_cleanup_on_exit=True)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        if cls._instance is not None:
            cls._instance.shutdown()
            cls._instance = None


def create_pipeline_graph() -> "CompiledGraph":
    """
    Create the pipeline graph (legacy API, backward compatible).

    Note: For proper resource management, prefer using GraphResources directly:

        with GraphResources() as resources:
            result = resources.graph.invoke({"text": "..."})

    Returns:
        Compiled LangGraph application
    """
    return GraphResources.get_instance().graph
