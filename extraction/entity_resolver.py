
import re
from collections import defaultdict
from copy import deepcopy


class EntityResolver:
    """
    Conservative entity resolution for enterprise knowledge graphs.

    Resolution strategies:
    1. Normalize whitespace and casing.
    2. Merge exact normalized duplicates of the same entity type.
    3. Resolve a single-token person alias to a full name only
       when that first name identifies exactly one full name.
    4. Never merge entities of different types.
    """

    def normalize(self, name: str) -> str:
        name = str(name or "").strip().lower()
        name = re.sub(r"\s+", " ", name)
        name = re.sub(r"[^\w\s.-]", "", name)
        return name.strip()

    def resolve(self, entities, relationships):
        entities = deepcopy(entities or [])
        relationships = deepcopy(relationships or [])

        # Group entities by type and normalized name.
        grouped = defaultdict(list)

        for entity in entities:
            name = entity.get("text", "").strip()
            entity_type = entity.get("label", "UNKNOWN")

            if not name:
                continue

            key = (entity_type, self.normalize(name))
            grouped[key].append(entity)

        canonical_entities = []
        alias_map = {}

        # Resolve exact duplicates first.
        for (entity_type, normalized_name), group in grouped.items():
            canonical = max(
                group,
                key=lambda item: len(item.get("text", "").strip())
            )

            canonical_name = canonical["text"].strip()

            canonical_entities.append(canonical)

            for entity in group:
                alias_map[entity["text"]] = canonical_name

        # Build first-name candidates for PERSON entities.
        full_names_by_first_name = defaultdict(set)

        for entity in canonical_entities:
            if entity.get("label") != "PERSON":
                continue

            name = entity["text"].strip()
            parts = name.split()

            if len(parts) > 1:
                first_name = self.normalize(parts[0])
                full_names_by_first_name[first_name].add(name)

        # Resolve short person aliases only when unambiguous.
        for entity in canonical_entities:
            if entity.get("label") != "PERSON":
                continue

            name = entity["text"].strip()
            parts = name.split()

            if len(parts) != 1:
                continue

            first_name = self.normalize(name)
            candidates = full_names_by_first_name.get(
                first_name, set()
            )

            if len(candidates) == 1:
                alias_map[name] = next(iter(candidates))

        # Update relationship endpoints using the alias map.
        for relationship in relationships:
            subject = relationship.get("subject")
            object_name = relationship.get("object")

            if subject:
                relationship["subject"] = alias_map.get(
                    subject, subject
                )

            if object_name:
                relationship["object"] = alias_map.get(
                    object_name, object_name
                )

        # Remove duplicate entities after alias resolution.
        resolved_entities = []
        seen = set()

        for entity in canonical_entities:
            original_name = entity["text"].strip()
            canonical_name = alias_map.get(
                original_name, original_name
            )

            key = (
                entity.get("label", "UNKNOWN"),
                self.normalize(canonical_name)
            )

            if key in seen:
                continue

            seen.add(key)

            updated_entity = deepcopy(entity)
            updated_entity["text"] = canonical_name
            resolved_entities.append(updated_entity)

        return {
            "entities": resolved_entities,
            "relationships": relationships,
            "alias_map": alias_map,
            "entities_before": len(entities),
            "entities_after": len(resolved_entities),
        }
