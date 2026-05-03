from __future__ import annotations

"""Generate interactive AOP pathway visualizations from Table 2 synthesis data."""

import json
from typing import Optional

import networkx as nx
import pandas as pd


# Color mapping for KE levels
KE_LEVEL_COLORS = {
    "MIE": "#e74c3c",                  # Red
    "Molecular": "#3498db",             # Blue
    "Cellular": "#2ecc71",              # Green
    "Tissue": "#f39c12",                # Orange
    "Organ": "#9b59b6",                 # Purple
    "Individual": "#1abc9c",            # Teal
    "Population": "#34495e",            # Dark gray
}

# Uncertainty level to node size mapping
UNCERTAINTY_SIZES = {
    "Low": 30,
    "Moderate": 25,
    "High": 20,
}


def build_pathway_graph(table2_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph representing the AOP pathway from Table 2 synthesis.

    Nodes represent Key Events (KEs), edges represent KERs (Key Event Relationships).
    Node attributes include KE level and uncertainty level for visualization.
    Edge attributes include confidence, contradiction info, and supporting paper count.

    Parameters
    ----------
    table2_df : pd.DataFrame
        Table 2 synthesis data from table2_synthesis.compute_table2().
        Expected columns: upstream_ke_name, downstream_ke_name, upstream_ke_level,
                         downstream_ke_level, ker_name, n_supporting_papers,
                         n_contradicting_papers, uncertainty_level, contradicts_ker, etc.

    Returns
    -------
    nx.DiGraph
        Directed graph with nodes and edges ready for visualization.
        Returns empty graph if table2_df is empty.
    """
    G = nx.DiGraph()

    if table2_df.empty:
        return G

    # Add nodes (Key Events) with metadata
    unique_kes = set()
    for _, row in table2_df.iterrows():
        up_ke = row.get("upstream_ke_name")
        down_ke = row.get("downstream_ke_name")
        if pd.notna(up_ke):
            unique_kes.add((str(up_ke).strip(), row.get("upstream_ke_level", "Molecular")))
        if pd.notna(down_ke):
            unique_kes.add((str(down_ke).strip(), row.get("downstream_ke_level", "Molecular")))

    for ke_name, ke_level in unique_kes:
        level_str = str(ke_level or "Molecular").strip()
        color = KE_LEVEL_COLORS.get(level_str, "#95a5a6")  # Gray fallback
        G.add_node(
            ke_name,
            level=level_str,
            color=color,
            size=25,  # Default size
        )

    # Add edges (KERs) with metadata
    for _, row in table2_df.iterrows():
        up_ke = str(row.get("upstream_ke_name", "")).strip()
        down_ke = str(row.get("downstream_ke_name", "")).strip()

        if not up_ke or not down_ke or up_ke not in G.nodes or down_ke not in G.nodes:
            continue

        ker_name = str(row.get("ker_name", f"{up_ke} → {down_ke}")).strip()
        n_supporting = int(row.get("n_supporting_papers", 0)) if pd.notna(row.get("n_supporting_papers")) else 0
        n_contra = int(row.get("n_contradicting_papers", 0)) if pd.notna(row.get("n_contradicting_papers")) else 0
        contradicts = bool(row.get("contradicts_ker", False))

        # Determine edge color: mostly supporting (green) or mostly contradicting (red)
        if n_supporting > 0 and n_contra == 0:
            edge_color = "#27ae60"  # Green
        elif n_contra > 0 and n_supporting == 0:
            edge_color = "#e74c3c"  # Red
        else:
            edge_color = "#f39c12"  # Orange (mixed)

        # Edge width proportional to number of supporting papers
        edge_width = min(10, max(2, 2 + n_supporting * 0.5))

        # Build tooltip/title
        tooltip = f"{ker_name}\n"
        tooltip += f"Supporting: {n_supporting} | Contradicting: {n_contra}"

        G.add_edge(
            up_ke,
            down_ke,
            title=tooltip,
            ker_name=ker_name,
            color=edge_color,
            width=edge_width,
            n_supporting=n_supporting,
            n_contradicting=n_contra,
            contradicts=contradicts,
        )

    return G


def render_interactive_graph(graph: nx.DiGraph, height: int = 800, physics: bool = True) -> str:
    """
    Render a networkx graph as an interactive HTML string using pyvis.

    The HTML is suitable for embedding in Streamlit via st.components.v1.html().

    Parameters
    ----------
    graph : nx.DiGraph
        The directed graph to visualize.
    height : int, optional
        Height of the visualization in pixels. Default 800.
    physics : bool, optional
        Whether to enable physics-based layout. Default True.

    Returns
    -------
    str
        HTML string containing the interactive visualization.
    """
    import json
    
    if graph.number_of_nodes() == 0:
        return "<p>No data to visualize. Add papers to Table 1 first.</p>"

    # Prepare nodes data
    nodes = []
    for node_id in graph.nodes():
        node_attrs = graph.nodes[node_id]
        nodes.append({
            "id": str(node_id),
            "label": str(node_id),
            "title": f"{node_id}\nLevel: {node_attrs.get('level', 'Unknown')}",
            "color": node_attrs.get("color", "#95a5a6"),
            "size": node_attrs.get("size", 25),
            "font": {"size": 16}
        })
    
    # Prepare edges data
    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append({
            "from": str(source),
            "to": str(target),
            "title": attrs.get("title", ""),
            "color": attrs.get("color", "#999999"),
            "width": attrs.get("width", 2),
            "arrows": "to"
        })
    
    # Physics configuration
    physics_config = {
        "enabled": physics,
        "stabilization": {"iterations": 200},
        "forceAtlas2Based": {
            "gravitationalConstant": -50,
            "centralGravity": 0.01,
            "springLength": 200
        },
        "timestep": 0.35
    } if physics else {"enabled": False}
    
    # Manually construct HTML using vis.js CDN
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <title>AOP Pathway Visualization</title>
        <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        <style type="text/css">
            html, body {{
                width: 100%;
                height: 100%;
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            #network {{
                width: 100%;
                height: 100%;
                border: 1px solid lightgray;
                background-color: #f9f9f9;
            }}
        </style>
    </head>
    <body>
        <div id="network"></div>
        <script type="text/javascript">
            var nodes = new vis.DataSet({json.dumps(nodes)});
            var edges = new vis.DataSet({json.dumps(edges)});
            var container = document.getElementById('network');
            var data = {{
                nodes: nodes,
                edges: edges
            }};
            var options = {{
                physics: {json.dumps(physics_config)},
                interaction: {{
                    navigationButtons: true,
                    keyboard: true
                }}
            }};
            var network = new vis.Network(container, data, options);
        </script>
    </body>
    </html>
    """
    
    return html


def export_graph_as_json(graph: nx.DiGraph) -> dict:
    """
    Export the graph as a JSON-compatible dictionary for external tools/sharing.

    Parameters
    ----------
    graph : nx.DiGraph
        The graph to export.

    Returns
    -------
    dict
        Dictionary with 'nodes' and 'edges' keys, each containing a list.
    """
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append({
            "id": node_id,
            "label": node_id,
            "level": attrs.get("level", "Unknown"),
            "color": attrs.get("color", "#95a5a6"),
            "size": attrs.get("size", 25),
        })

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append({
            "source": source,
            "target": target,
            "label": attrs.get("ker_name", ""),
            "n_supporting": attrs.get("n_supporting", 0),
            "n_contradicting": attrs.get("n_contradicting", 0),
            "color": attrs.get("color", "#999999"),
            "width": attrs.get("width", 2),
        })

    return {"nodes": nodes, "edges": edges}


def get_pathway_chains(graph: nx.DiGraph, max_length: int = 10) -> list[list[str]]:
    """
    Extract all simple paths (chains) from the graph up to a maximum length.

    Useful for identifying complete mechanistic pathways.

    Parameters
    ----------
    graph : nx.DiGraph
        The pathway graph.
    max_length : int, optional
        Maximum length of paths to return. Default 10.

    Returns
    -------
    list[list[str]]
        List of paths, each path is a list of node IDs.
    """
    if graph.number_of_nodes() == 0:
        return []

    # Find all simple paths in the graph
    all_paths = []
    source_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]

    for source in source_nodes:
        try:
            for target in graph.nodes():
                try:
                    paths = list(nx.all_simple_paths(graph, source, target, cutoff=max_length))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    pass
        except Exception:
            pass

    # Deduplicate and sort by length (longest first)
    unique_paths = list(set(tuple(p) for p in all_paths))
    unique_paths = sorted(unique_paths, key=len, reverse=True)

    return [list(p) for p in unique_paths]
