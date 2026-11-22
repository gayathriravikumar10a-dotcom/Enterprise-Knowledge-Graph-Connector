import re
from collections import defaultdict
from copy import deepcopy


STOP_WORDS = {"and", "of", "the", "for", "in", "on"}

TYPE_SUFFIXES = {
    "DEPARTMENT": {"department", "dept", "division", "unit"},
    "TEAM": {"team", "group", "squad", "pod"},
    "ORG": {"inc", "incorporated", "corp", "corporation", "ltd", "limited", "llc", "co", "company"},
}


class EntityResolver:
    """
    Enterprise entity resolution for knowledge graphs.

    Resolution strategies:
    1. Normalize whitespace, punctuation, and casing.
    2. Merge exact normalized duplicates of the same entity type.
    3. Merge suffix variants of the same type (e.g., 'Technology' -> 'Technology Department').
    4. Merge unambiguous acronyms of the same type (e.g., 'AI' -> 'Artificial Intelligence').
    5. Resolve unambiguous single-token first names ('Ravi' -> 'Ravi Sharma') and
       first-initial + last-name aliases ('R. Sharma' -> 'Ravi Sharma') for PERSON entities.
    6. Never merge entities of different types.
    """

    def normalize(self, name: str) -> str:
        name = str(name or "").strip().lower()
        name = re.sub(r"\s+", " ", name)
        name = re.sub(r"[^\w\s.-]", "", name)
        return name.strip()

    def _strip_type_suffix(self, normalized_name: str, entity_type: str) -> str:
        suffixes = TYPE_SUFFIXES.get(entity_type, set())
        if not suffixes:
            return normalized_name
        tokens = [t.rstrip(".") for t in normalized_name.split()]
        while len(tokens) > 1 and tokens[-1] in suffixes:
            tokens.pop()
        return " ".join(tokens)

    def _acronym(self, name: str) -> str:
        tokens = [
            re.sub(r"[^\w]", "", t)
            for t in str(name or "").strip().split()
        ]
        meaningful = [t for t in tokens if t and t.lower() not in STOP_WORDS]
        if len(meaningful) < 2:
            return ""
        return "".join(t[0].lower() for t in meaningful)

    def resolve(self, entities, relationships):
        entities = deepcopy(entities or [])
        relationships = deepcopy(relationships or [])

        # 1. Group entities by type and exact normalized name.
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

        for (entity_type, _), group in grouped.items():
            canonical = max(
                group,
                key=lambda item: len(item.get("text", "").strip())
            )
            canonical_name = canonical["text"].strip()
            canonical_entities.append(canonical)

            for entity in group:
                alias_map[entity["text"]] = canonical_name

        # 2. Merge organizational suffix variants within the same entity type.
        by_type_core = defaultdict(list)
        for entity in canonical_entities:
            etype = entity.get("label", "UNKNOWN")
            if etype not in TYPE_SUFFIXES:
                continue
            name = entity["text"].strip()
            core = self._strip_type_suffix(self.normalize(name), etype)
            by_type_core[(etype, core)].append(name)

        for (etype, core), names in by_type_core.items():
            if len(names) > 1:
                preferred = max(names, key=len)
                for n in names:
                    alias_map[n] = preferred

        # 3. Merge unambiguous acronyms within the same entity type.
        acronym_candidates = defaultdict(set)
        for entity in canonical_entities:
            etype = entity.get("label", "UNKNOWN")
            current_name = alias_map.get(entity["text"].strip(), entity["text"].strip())
            acr = self._acronym(current_name)
            if acr:
                acronym_candidates[(etype, acr)].add(current_name)

        for entity in canonical_entities:
            etype = entity.get("label", "UNKNOWN")
            name = entity["text"].strip()
            norm = re.sub(r"[^\w]", "", self.normalize(name))
            if len(norm) >= 2 and len(name.split()) == 1:
                candidates = acronym_candidates.get((etype, norm), set())
                if len(candidates) == 1:
                    alias_map[name] = next(iter(candidates))

        # 4. Resolve PERSON first-name and first-initial + last-name aliases when unambiguous.
        full_names_by_first_name = defaultdict(set)
        full_names_by_initial_last = defaultdict(set)

        for entity in canonical_entities:
            if entity.get("label") != "PERSON":
                continue

            name = alias_map.get(entity["text"].strip(), entity["text"].strip())
            parts = name.split()

            if len(parts) > 1:
                first_clean = self.normalize(parts[0]).rstrip(".")
                last_clean = self.normalize(parts[-1]).rstrip(".")
                if len(first_clean) > 1:
                    full_names_by_first_name[first_clean].add(name)
                    full_names_by_initial_last[(first_clean[0], last_clean)].add(name)

        for entity in canonical_entities:
            if entity.get("label") != "PERSON":
                continue

            name = entity["text"].strip()
            parts = name.split()

            if len(parts) == 1:
                first_name = self.normalize(name).rstrip(".")
                candidates = full_names_by_first_name.get(first_name, set())
                if len(candidates) == 1:
                    alias_map[name] = next(iter(candidates))
            elif len(parts) == 2:
                first_token = self.normalize(parts[0]).rstrip(".")
                last_token = self.normalize(parts[1]).rstrip(".")
                if len(first_token) == 1:
                    candidates = full_names_by_initial_last.get(
                        (first_token, last_token), set()
                    )
                    if len(candidates) == 1:
                        alias_map[name] = next(iter(candidates))

        # Propagate transitive alias mappings (A -> B -> C becomes A -> C).
        for key in list(alias_map.keys()):
            target = alias_map[key]
            visited = {key}
            while target in alias_map and alias_map[target] != target and target not in visited:
                visited.add(target)
                target = alias_map[target]
            alias_map[key] = target

        # 5. Update and deduplicate relationships using canonical names.
        resolved_relationships = []
        seen_rels = set()

        for relationship in relationships:
            subject = relationship.get("subject")
            object_name = relationship.get("object")

            if subject:
                relationship["subject"] = alias_map.get(subject, subject)
            if object_name:
                relationship["object"] = alias_map.get(object_name, object_name)

            rel_key = (
                relationship.get("subject"),
                relationship.get("predicate"),
                relationship.get("object"),
            )
            if rel_key in seen_rels:
                continue
            seen_rels.add(rel_key)
            resolved_relationships.append(relationship)

        # 6. Deduplicate canonical entities and record merged surface aliases.
        aliases_by_canonical = defaultdict(set)
        for surface, canonical in alias_map.items():
            if surface != canonical:
                aliases_by_canonical[canonical].add(surface)

        resolved_entities = []
        seen = set()

        for entity in canonical_entities:
            original_name = entity["text"].strip()
            canonical_name = alias_map.get(original_name, original_name)

            key = (
                entity.get("label", "UNKNOWN"),
                self.normalize(canonical_name)
            )

            if key in seen:
                continue

            seen.add(key)

            updated_entity = deepcopy(entity)
            updated_entity["text"] = canonical_name
            updated_entity["aliases"] = sorted(aliases_by_canonical.get(canonical_name, set()))
            resolved_entities.append(updated_entity)

        return {
            "entities": resolved_entities,
            "relationships": resolved_relationships,
            "alias_map": alias_map,
            "entities_before": len(entities),
            "entities_after": len(resolved_entities),
        }