"""
LangGraph Pipeline Graph with resource lifecycle management.
"""

from __future__ import annotations

import logging
import atexit
from typing import Optional, TYPE_CHECKING

from langgraph.graph import StateGraph, END
from src.state import PipelineState
from src.nodes.detection.detection_node import DetectionNode
from src.nodes.anonymisation.anonymization_node import AnonymizationNode

if TYPE_CHECKING:
    from langgraph.graph.graph import CompiledGraph

logger = logging.getLogger("PipeGraph")


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

        # 3. Add nodes to graph
        workflow.add_node("detection", self._detection_node)
        workflow.add_node("anonymization", self._anonymization_node)

        # 4. Define edges (flow)
        workflow.set_entry_point("detection")
        workflow.add_edge("detection", "anonymization")
        workflow.add_edge("anonymization", END)

        # 5. Compile
        logger.info("Pipeline graph compiled")
        return workflow.compile()

    def shutdown(self, wait: bool = True) -> None:
        """
        Clean up all resources.

        Args:
            wait: If True, wait for executor tasks to complete
        """
        logger.info("Shutting down pipeline resources...")

        # Shutdown shared executor
        try:
            DetectionNode.shutdown_executor(wait=wait)
        except Exception as e:
            logger.warning(f"Error shutting down executor: {e}")

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
