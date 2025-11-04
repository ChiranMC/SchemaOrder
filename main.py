from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Any, Dict, List, Tuple, Optional, Set
import json, re, heapq
from collections import defaultdict

app = FastAPI(title="Schema Dependency Analyzer", version="1.1")

# ----------------- Utility Functions ----------------- #

def clean_identifier(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return raw
    quoted = re.findall(r'"([^"]+)"', raw)
    if quoted:
        return quoted[-1]
    raw = raw.split('.')[-1]
    return raw.strip('"')

def parse_reference_string(ref: Optional[str]) -> Tuple[Optional[str], List[str]]:
    if not ref:
        return None, []
    match = re.match(r'\s*([^(]+?)(?:\s*\(\s*(.*?)\s*\))?\s*$', ref)
    if not match:
        return clean_identifier(ref), []
    table = clean_identifier(match.group(1))
    columns = match.group(2)
    if not columns:
        return table, []
    column_list = [col.strip().strip('"') for col in columns.split(',') if col.strip()]
    return table, column_list

def load_tables_from_dict(raw_tables: list) -> Dict[str, dict]:
    tables: Dict[str, dict] = {}
    for entry in raw_tables:
        name = entry["name"]
        info = {
            "name": name,
            "primary_keys": set(entry.get("primary_keys") or []),
            "foreign_keys": [],
            "dependencies": set(entry.get("dependencies") or []),
        }

        for fk_entry in entry.get("foreign_keys", []):
            columns = fk_entry.get("columns")
            if columns is None and fk_entry.get("column") is not None:
                columns = [fk_entry["column"]]
            columns = columns or []
            ref_table, ref_columns = parse_reference_string(fk_entry.get("references"))
            fk_struct = {
                "columns": columns,
                "column": columns[0] if len(columns) == 1 else columns,
                "referenced_table": ref_table,
                "referenced_columns": ref_columns,
                "references": fk_entry.get("references"),
                "constraint_name": fk_entry.get("constraint_name"),
            }
            info["foreign_keys"].append(fk_struct)
            if ref_table:
                info["dependencies"].add(ref_table)

        tables[name] = info
    return tables

def strongly_connected_components(
    tables: Dict[str, dict]
) -> Tuple[List[List[str]], Dict[str, int], Set[str]]:
    nodes = sorted(tables.keys())
    adjacency: Dict[str, Set[str]] = {
        table: {dep for dep in info["dependencies"] if dep in tables and dep != table}
        for table, info in tables.items()
    }
    self_loops = {table for table, info in tables.items() if table in info["dependencies"]}

    index = 0
    indices: Dict[str, int] = {}
    lowlinks: Dict[str, int] = {}
    stack: List[str] = []
    on_stack: Set[str] = set()
    components_raw: List[List[str]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in adjacency[node]:
            if neighbor not in indices:
                strongconnect(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])

        if lowlinks[node] == indices[node]:
            component = []
            while True:
                neighbor = stack.pop()
                on_stack.remove(neighbor)
                component.append(neighbor)
                if neighbor == node:
                    break
            components_raw.append(component)

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    components = [sorted(component) for component in components_raw]
    component_map: Dict[str, int] = {}
    for idx, component in enumerate(components):
        for table in component:
            component_map[table] = idx
    return components, component_map, self_loops

def topological_component_order(
    tables: Dict[str, dict],
    components: List[List[str]],
    component_map: Dict[str, int]
) -> List[int]:
    comp_edges: Dict[int, Set[int]] = defaultdict(set)
    indegree: Dict[int, int] = {idx: 0 for idx in range(len(components))}

    for table, info in tables.items():
        dst = component_map[table]
        for dependency in info["dependencies"]:
            if dependency not in tables:
                continue
            src = component_map[dependency]
            if src == dst:
                continue
            if dst not in comp_edges[src]:
                comp_edges[src].add(dst)
                indegree[dst] += 1

    heap: List[Tuple[str, int]] = []
    for idx, deg in indegree.items():
        if deg == 0:
            first_name = components[idx][0] if components[idx] else ""
            heapq.heappush(heap, (first_name, idx))

    order: List[int] = []
    while heap:
        _, comp_idx = heapq.heappop(heap)
        order.append(comp_idx)
        for neighbor in sorted(comp_edges[comp_idx]):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                first_name = components[neighbor][0] if components[neighbor] else ""
                heapq.heappush(heap, (first_name, neighbor))

    if len(order) != len(components):
        missing = set(range(len(components))) - set(order)
        raise RuntimeError(f"Topological sorting failed for component indices: {missing}")
    return order

def build_output_from_json(raw_tables: list) -> Dict[str, Any]:
    tables = load_tables_from_dict(raw_tables)

    for info in tables.values():
        info["foreign_keys"] = sorted(
            info["foreign_keys"],
            key=lambda fk: (fk["referenced_table"] or "", fk["columns"]),
        )
        info["dependencies"] = sorted(info["dependencies"])

    components, component_map, self_loops = strongly_connected_components(tables)
    component_order = topological_component_order(tables, components, component_map)
    topological_groups = [components[idx] for idx in component_order]
    topological_order = [table for group in topological_groups for table in group]

    cycles = [group for group in topological_groups if len(group) > 1]
    self_referencing_tables = sorted(
        table for table in topological_order if table in self_loops
    )

    result_tables = []
    for name, info in sorted(tables.items()):
        dependencies = [dep for dep in info["dependencies"] if dep != name]
        self_deps = [name] if name in info["dependencies"] else []
        result_tables.append(
            {
                "name": name,
                "primary_keys": sorted(info["primary_keys"]),
                "foreign_keys": info["foreign_keys"],
                "dependencies": dependencies,
                "self_dependencies": self_deps,
                "component": component_map[name],
            }
        )

    output = {
        "tables": result_tables,
        "topological_groups": topological_groups,
        "topological_order": topological_order,
    }
    if cycles:
        output["cycles"] = cycles
    if self_referencing_tables:
        output["self_referencing_tables"] = self_referencing_tables
    return output

# ----------------- FastAPI Models & Endpoints ----------------- #

class SchemaBody(BaseModel):
    schema: list  # List of tables (same structure as your JSON file)

@app.get("/")
def root():
    return {"message": "Schema Dependency Analyzer API (JSON body version)"}

@app.post("/analyze")
def analyze_schema(body: SchemaBody):
    try:
        result = build_output_from_json(body.schema)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
