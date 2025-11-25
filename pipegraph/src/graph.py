from langgraph.graph import StateGraph, END
from src.state import PipelineState
from src.nodes.detection.detection_node import DetectionNode
from src.nodes.anonymisation.anonymization_node import AnonymizationNode

def create_pipeline_graph():
    # 1. Initialisation du graphe avec notre State typé
    workflow = StateGraph(PipelineState)

    # 2. Instanciation des nodes
    detection_node = DetectionNode() # Hybrid Detection (Deterministic + AI)
    anonymization_node = AnonymizationNode()

    # 3. Ajout des nodes au graphe
    workflow.add_node("detection", detection_node)
    workflow.add_node("anonymization", anonymization_node)

    # 4. Définition des edges (flux)
    # Entrypoint -> Detection
    workflow.set_entry_point("detection")
    
    # Detection -> Anonymization
    workflow.add_edge("detection", "anonymization")
    
    # Anonymization -> End
    workflow.add_edge("anonymization", END)

    # 5. Compilation
    app = workflow.compile()
    return app
