from collections import deque


class GraphReasoner:

    def __init__(self, knowledge_graph):
        self.kg = knowledge_graph

    def find_objects(self, subject, predicate):
        results = []

        if subject not in self.kg.graph:
            return results

        for _, target, data in self.kg.graph.out_edges(
            subject, data=True
        ):
            if data.get("relation") == predicate:
                results.append(target)

        return results

    def find_subjects(self, predicate, object_name):
        results = []

        if object_name not in self.kg.graph:
            return results

        for source, _, data in self.kg.graph.in_edges(
            object_name, data=True
        ):
            if data.get("relation") == predicate:
                results.append(source)

        return results

    def find_paths(self, start_entity: str, end_entity: str, max_hops: int = 3) -> list[dict]:
        """
        Find directed multi-hop reasoning paths between two entities up to max_hops.
        Returns structured paths with human-readable explanations and evidence.
        """
        graph = self.kg.graph
        if start_entity not in graph or end_entity not in graph:
            return []

        if start_entity == end_entity:
            return []

        paths = []
        queue = deque([(start_entity, [], {start_entity})])

        while queue:
            current, steps, visited = queue.popleft()

            if len(steps) >= max_hops:
                continue

            for _, neighbor, data in graph.out_edges(current, data=True):
                step = {
                    "subject": current,
                    "predicate": data.get("relation", "RELATED_TO"),
                    "object": neighbor,
                    "evidence": data.get("evidence", ""),
                }
                new_steps = steps + [step]

                if neighbor == end_entity:
                    paths.append({
                        "start": start_entity,
                        "end": end_entity,
                        "hops": len(new_steps),
                        "steps": new_steps,
                        "reasoning": self._format_path_reasoning(new_steps),
                    })
                elif neighbor not in visited:
                    queue.append((neighbor, new_steps, visited | {neighbor}))

        return paths

    def get_entity_context(self, entity_name: str, max_hops: int = 2) -> list[dict]:
        """
        Traverse up to max_hops outward and 1-hop inward around an entity
        to gather connected enterprise context for Hybrid RAG retrieval.
        """
        graph = self.kg.graph
        if entity_name not in graph:
            return []

        discovered = []
        seen_edges = set()

        # Incoming 1-hop relationships (e.g., who manages or belongs to this entity)
        for source, target, data in graph.in_edges(entity_name, data=True):
            predicate = data.get("relation", "RELATED_TO")
            edge_key = (source, predicate, target)
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                discovered.append({
                    "subject": source,
                    "predicate": predicate,
                    "object": target,
                    "hop": 1,
                    "direction": "incoming",
                    "evidence": data.get("evidence", ""),
                })

        # Outgoing multi-hop traversal (e.g., Person -> Team -> Department / Project)
        queue = deque([(entity_name, 0)])
        visited = {entity_name}

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue

            for source, target, data in graph.out_edges(current, data=True):
                predicate = data.get("relation", "RELATED_TO")
                edge_key = (source, predicate, target)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    discovered.append({
                        "subject": source,
                        "predicate": predicate,
                        "object": target,
                        "hop": depth + 1,
                        "direction": "outgoing",
                        "evidence": data.get("evidence", ""),
                    })

                if target not in visited:
                    visited.add(target)
                    queue.append((target, depth + 1))

        return discovered

    def infer_indirect_departments(self, entity_name: str) -> list[dict]:
        """
        Infer department membership either directly (Entity -BELONGS_TO-> Dept)
        or via a team/project (Entity -BELONGS_TO/WORKS_ON-> Mid -BELONGS_TO-> Dept).
        """
        graph = self.kg.graph
        if entity_name not in graph:
            return []

        results = []
        seen = set()

        # Direct department
        for dept in self.find_objects(entity_name, "BELONGS_TO"):
            if graph.nodes[dept].get("entity_type") == "DEPARTMENT" and dept not in seen:
                seen.add(dept)
                results.append({
                    "entity": entity_name,
                    "department": dept,
                    "hops": 1,
                    "reasoning": f"{entity_name} belongs directly to {dept}.",
                })

        # 2-hop via TEAM or PROJECT
        for rel in ("BELONGS_TO", "WORKS_ON"):
            for intermediate in self.find_objects(entity_name, rel):
                for dept in self.find_objects(intermediate, "BELONGS_TO"):
                    if graph.nodes[dept].get("entity_type") == "DEPARTMENT" and dept not in seen:
                        seen.add(dept)
                        results.append({
                            "entity": entity_name,
                            "via": intermediate,
                            "department": dept,
                            "hops": 2,
                            "reasoning": (
                                f"{entity_name} ({rel}) {intermediate}, "
                                f"which belongs to {dept}."
                            ),
                        })

        return results

    def who_manages_persons_project(self, person):
        projects = list(self.find_objects(person, "WORKS_ON"))

        # Also include projects worked on by teams the person belongs to
        for team in self.find_objects(person, "BELONGS_TO"):
            for proj in self.find_objects(team, "WORKS_ON"):
                if proj not in projects:
                    projects.append(proj)

        answers = []

        for project in projects:
            managers = self.find_subjects("MANAGES", project)

            for manager in managers:
                answers.append({
                    "person": person,
                    "project": project,
                    "manager": manager,
                    "reasoning": (
                        f"{person} works on {project}, "
                        f"which is managed by {manager}."
                    )
                })

        return answers

    def explain_relationship(self, subject, predicate, object_name):
        if subject not in self.kg.graph:
            return False

        for _, target, data in self.kg.graph.out_edges(
            subject, data=True
        ):
            if (
                target == object_name
                and data.get("relation") == predicate
            ):
                return data.get("evidence", "")

        return False

    @staticmethod
    def _format_path_reasoning(steps: list[dict]) -> str:
        parts = []
        for idx, step in enumerate(steps):
            if idx == 0:
                parts.append(f"{step['subject']} --[{step['predicate']}]--> {step['object']}")
            else:
                parts.append(f"--[{step['predicate']}]--> {step['object']}")
        return " ".join(parts)