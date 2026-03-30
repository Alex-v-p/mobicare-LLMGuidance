from __future__ import annotations

from typing import Callable


FamilyQueryOrder = Callable[[], tuple[str, ...]]
FamilyQueryTemplate = Callable[[str], str]
DefaultAgentResolver = Callable[[str], str]


AGENT_KEYS = {
    "mra": "mra_agent",
    "beta_blocker": "beta_blocker_agent",
    "ras": "ras_agent",
    "sglt2": "sglt2_agent",
    "loop_diuretic": "loop_agent",
}


def build_drug_retrieval_queries(
    snapshot: dict[str, object],
    *,
    family_query_order: FamilyQueryOrder,
    family_query_template: FamilyQueryTemplate,
    default_agent: DefaultAgentResolver,
) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    for family in family_query_order():
        template = family_query_template(family)
        if not template:
            continue
        agent = snapshot.get(AGENT_KEYS.get(family, "")) if family in AGENT_KEYS else None
        resolved_agent = str(agent or default_agent(family))
        query = template.format(agent=resolved_agent) if "{agent}" in template else template
        queries.append({"family": family, "query": query})
    return queries
