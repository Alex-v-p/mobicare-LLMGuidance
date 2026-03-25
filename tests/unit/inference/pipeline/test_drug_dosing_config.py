from inference.pipeline.support import drug_dosing



def test_build_drug_retrieval_queries_reads_external_catalog(monkeypatch):
    monkeypatch.setattr(
        drug_dosing,
        "load_drug_dosing_catalog_payload",
        lambda: {
            "default_agents": {"beta_blocker": "carvedilol"},
            "family_query_order": ["beta_blocker"],
            "family_priority": {"beta_blocker": 1},
            "families": {
                "beta_blocker": {
                    "keywords": ["beta blocker"],
                    "query_template": "{agent} beta-blocker custom query",
                }
            },
        },
    )

    queries = drug_dosing.build_drug_retrieval_queries({"beta_blocker_agent": None})

    assert queries == [{"family": "beta_blocker", "query": "carvedilol beta-blocker custom query"}]
