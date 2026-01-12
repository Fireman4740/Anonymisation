# Anonymisation Project Instructions

You are working on an advanced text anonymization system with a hybrid architecture (Regex + NER + LLM). The project is currently in a transition phase between a legacy pipeline and a new LangGraph-based implementation.

## 🏗 Project Structure & Architecture

The codebase is divided into three main components:

1.  **Legacy Pipeline (`pipeline/`)**: The current stable implementation.
    -   **Core**: `src/core/orchestrator.py` manages the flow.
    -   **Layers**: Detection (Regex/NER), Transformation (Anonymization/Generalization), Evaluation.
    -   **API**: `scripts/api_server.py` exposes the functionality via FastAPI.

2.  **Next-Gen Pipeline (`pipegraph/`)**: The future implementation using LangGraph.
    -   **Architecture**: StateGraph with modular nodes (`DetectionNode`, `AnonymizationNode`).
    -   **State**: `PipelineState` carries text, entities, and config through the graph.
    -   **Entry**: `main.py` and `src/graph.py`.

3.  **Evaluation & Tools (`eval/`)**:
    -   **Benchmarks**: `benchmark_pipeline.py` measures performance against datasets (`DB-bio`, `PersonalReddit`, `TAB`).
    -   **Visualization**: `streamlit_app/app.py` for analyzing errors (TP/FP/FN) and reports.

## 🚀 Critical Workflows

-   **Run Legacy API**:
    ```bash
    python pipeline/scripts/api_server.py
    ```
-   **Run New Pipeline (PipeGraph)**:
    ```bash
    python pipegraph/main.py
    ```
-   **Run Benchmarks**:
    ```bash
    python eval/benchmark_pipeline.py --dataset eval/datasets/data/anonymization_dataset.json --limit 50
    ```
-   **Run Visualization**:
    ```bash
    streamlit run eval/streamlit_app/app.py
    ```
-   **Run Tests**:
    ```bash
    pytest pipeline/tests/
    ```

## 💻 Coding Conventions

-   **Python Version**: 3.11+
-   **Type Hinting**: Strictly use type hints (`typing.List`, `typing.Optional`, etc.) for all function signatures.
-   **Testing Pattern**: Use **Stubs** (e.g., `StubDetectionService`) instead of mocks for service isolation in unit tests. See `pipeline/tests/test_pipeline.py`.
-   **Path Management**: Use `os.path.abspath` and `sys.path` manipulation in scripts to ensure modules are importable from the project root.
-   **LangGraph**: When working in `pipegraph/`, ensure nodes are pure functions or classes that modify the `PipelineState` and return a dictionary of updates.

## ⚠️ Important Context

-   **Transition**: We are moving logic from `pipeline/` to `pipegraph/`. When asked to implement features, clarify if they apply to the legacy or new pipeline.
-   **Evaluation**: The `eval/` folder is central to validating changes. Always consider how a change affects the metrics (Precision/Recall) on the datasets.
-   **Dependencies**: Each folder (`pipeline`, `pipegraph`, `eval`) has its own `requirements.txt`. Ensure you are using the correct environment.
