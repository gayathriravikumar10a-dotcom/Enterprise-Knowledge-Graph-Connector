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

    def who_manages_persons_project(self, person):
        projects = self.find_objects(person, "WORKS_ON")

        answers = []

        for project in projects:
            managers = self.find_subjects(project and "MANAGES", project)

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
