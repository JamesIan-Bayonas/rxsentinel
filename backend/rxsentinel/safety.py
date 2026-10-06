from collections import defaultdict
from itertools import combinations

import networkx as nx

from rxsentinel.catalog import Catalog
from rxsentinel.schemas import (
    AuditReport,
    AuditRequest,
    ExcludedEntry,
    Finding,
    Product,
    RuleSet,
)


class SafetyEngine:
    def __init__(self, catalog: Catalog, rules: RuleSet):
        self.catalog = catalog
        self.rules = rules
        self.graph = nx.MultiGraph()
        for rule in rules.rules:
            self.graph.add_edge(
                rule.ingredient_a, rule.ingredient_b, key=rule.rule_id, rule=rule
            )

    def audit(self, request: AuditRequest) -> AuditReport:
        included: dict[str, Product] = {}
        excluded: list[ExcludedEntry] = []
        for entry in request.medications:
            product = self.catalog.get(entry.product_id)
            if not entry.identity_confirmed:
                reason = "identity_unconfirmed"
            elif product is None:
                reason = "unknown_product"
            elif any(not ingredient.rxcui for ingredient in product.ingredients):
                reason = "unresolved_ingredients"
            else:
                included[entry.entry_id] = product
                continue
            excluded.append(ExcludedEntry(entry_id=entry.entry_id, reason=reason))

        by_ingredient: dict[str, set[str]] = defaultdict(set)
        for entry_id, product in included.items():
            for ingredient in product.ingredients:
                if ingredient.rxcui:
                    by_ingredient[ingredient.rxcui].add(entry_id)

        findings: list[Finding] = []
        for rxcui, entry_ids in sorted(by_ingredient.items()):
            if len(entry_ids) < 2:
                continue
            sources = [included[e].source for e in sorted(entry_ids)]
            sources.extend(
                ingredient.normalization_source
                for entry_id in sorted(entry_ids)
                for ingredient in included[entry_id].ingredients
                if ingredient.rxcui == rxcui and ingredient.normalization_source is not None
            )
            findings.append(
                Finding(
                    kind="possible_duplicate_ingredient",
                    title="Medication entries share an active ingredient",
                    description="Separate reviewed medication entries map to the same RxNorm "
                    "ingredient. This does not establish an overdose or an unintended regimen.",
                    recommendation="Review the entries against the prescribed medication list.",
                    entry_ids=sorted(entry_ids),
                    ingredient_rxcuis=[rxcui],
                    sources=sources,
                )
            )

        unsupported: list[tuple[str, str]] = []
        pairs = list(combinations(sorted(by_ingredient), 2))
        for a, b in pairs:
            if not self.graph.has_edge(a, b):
                unsupported.append((a, b))
                continue
            for _, attributes in sorted(self.graph[a][b].items()):
                rule = attributes["rule"]
                findings.append(
                    Finding(
                        kind="documented_interaction",
                        title=rule.title,
                        description=rule.description,
                        recommendation=rule.recommendation,
                        entry_ids=sorted(by_ingredient[a] | by_ingredient[b]),
                        ingredient_rxcuis=[a, b],
                        sources=[rule.source],
                        rule_id=rule.rule_id,
                        validation_status=rule.validation_status,
                    )
                )

        status = "findings_require_review" if findings else "no_matching_findings"
        if not included:
            status = "insufficient_data"
        limitations = [
            "Research prototype; rule coverage is limited and not a comprehensive clinical audit.",
            "Absence of a matching rule does not establish medication safety.",
            "Dose, schedule, patient history, allergies, and organ function are not assessed.",
            "Reviewed identities are user assertions; the system does not independently verify them.",
        ]
        if excluded:
            limitations.append("Some medication entries were excluded; the assessment is incomplete.")
        return AuditReport(
            status=status,
            rule_set_version=self.rules.version,
            catalog_snapshots=sorted({p.snapshot_id for p in included.values()}),
            included_entry_ids=sorted(included),
            excluded_entries=excluded,
            findings=findings,
            evaluated_distinct_ingredient_pairs=len(pairs),
            pairs_without_documented_rules=unsupported,
            interaction_coverage=self.rules.coverage,
            limitations=limitations,
        )
