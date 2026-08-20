from __future__ import annotations

"""
A claim a person typed has to behave like every other claim, and be labelled
like none of them.

Those two requirements pull against each other, and each half of the tension
has already been a bug in a tool of this shape. Store manual rows somewhere of
their own and the pathway does not see them: normalization rebuilds canonical
Key Events from Table 1, and a KER is a group-by over Table 1, so a claim that
is not a Table 1 row is a claim that exists on screen and nowhere in the graph.
Store them identically and the tool's central promise fails the other way — the
QC report reports a curator's typing as the model's accuracy, and the figure
draws an assertion in the same green as three concordant studies.

So the tests below come in two halves: the row is ordinary enough to flow all
the way to the map, and distinguishable enough that nothing downstream can
mistake it for evidence.
"""

import pytest

from schemas import KE_LEVEL_ORDER


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A scratch database with one extracted row already in it."""
    import stage2_extraction.table1_store as ts

    monkeypatch.setattr(ts, "DB_PATH", tmp_path / "scratch.db")
    ts.init_db()

    from schemas import EvidenceSpan, KERExtraction

    extracted = KERExtraction(
        upstream_ke_name="Decreased sodium-channel activity",
        upstream_ke_level="Molecular",
        downstream_ke_name="Impaired oligodendrocyte differentiation",
        downstream_ke_level="Cellular",
        ker_name="Sodium-channel loss impairs differentiation",
        ker_description="From the model.",
        ker_adjacency="Adjacent",
        paper_type="Primary study",
        cited_evidence_dois=None,
        biological_plausibility=None,
        empirical_evidence_summary=None,
        essentiality_evidence=None,
        contradicts_ker=False,
        taxonomic_applicability="rat",
        sex_applicability="both",
        life_stage_applicability="juvenile",
        modulating_factors=None,
        quantitative_relationships=None,
        response_response_relationship=None,
        time_scale=None,
        feedforward_feedback_loops=None,
        study_design="in vivo",
        exposure_route=None,
        chemical_stressor=None,
        extraction_confidence="High",
    )
    extracted.evidence_spans = [
        EvidenceSpan(quote="A located sentence.", field="ker_description",
                     verified=True, match_ratio=1.0)
    ]
    ts.insert_table1_row(
        extracted, "10.1234/extracted", {}, source_filename="paper.pdf"
    )
    return ts


def _claim(**overrides):
    values = {
        "upstream_ke_name": "Impaired oligodendrocyte differentiation",
        "upstream_ke_level": "Cellular",
        "downstream_ke_name": "Reduced myelination",
        "downstream_ke_level": "Tissue",
        "direction": "positive",
        "relation_kind": "causal",
        "ker_adjacency": "Adjacent",
        "entry_rationale": "Stated in the Figure 4 legend; extraction read "
                           "only the body text.",
        "source_doi": "10.1234/extracted",
    }
    values.update(overrides)
    return values


class TestItIsAnOrdinaryRow:
    """The half that makes a manual claim reach the map at all."""

    def test_a_manual_claim_lands_in_table_1(self, store):
        from stage2_extraction import manual_entry

        manual_entry.save_manual_claim(_claim(), curator="julia")

        rows = store.load_table1_as_dataframe()
        assert len(rows) == 2, "the claim has to be a Table 1 row, not a side table"
        assert "Reduced myelination" in set(rows["downstream_ke_name"])

    def test_it_normalizes_into_a_canonical_key_event(self, store):
        """
        The point of storing it as a row: normalization picks it up unaided.

        If this fails, a curator can enter a claim and never see it on the map,
        which is indistinguishable from the extraction having missed it.
        """
        from stage2_extraction import ke_normalizer, manual_entry

        manual_entry.save_manual_claim(_claim(), curator="julia")
        ke_normalizer.normalize_table1(
            store.load_table1_as_dataframe(), threshold=0.86, ols4_enabled=False
        )

        names = {
            str(n).casefold()
            for n in store.load_canonical_kes()["canonical_name"]
        }
        assert "reduced myelination" in names

    def test_it_produces_a_relationship_between_canonical_events(self, store):
        """A KER is a group-by over Table 1, so the row has to close the pair."""
        from stage2_extraction import ke_normalizer, manual_entry

        manual_entry.save_manual_claim(_claim(), curator="julia")
        ke_normalizer.normalize_table1(
            store.load_table1_as_dataframe(), threshold=0.86, ols4_enabled=False
        )

        rows = store.load_table1_as_dataframe()
        manual = rows[rows["origin"] == "curator"].iloc[0]
        assert manual["upstream_ke_canonical_id"] is not None
        assert manual["downstream_ke_canonical_id"] is not None


class TestItIsNotMistakenForEvidence:
    """The half that keeps the tool's promise."""

    def test_the_row_records_who_entered_it_and_why(self, store):
        from stage2_extraction import manual_entry

        manual_entry.save_manual_claim(_claim(), curator="julia")
        row = store.load_table1_as_dataframe().iloc[-1]

        assert row["origin"] == "curator"
        assert row["entered_by"] == "julia"
        assert "Figure 4" in str(row["entry_rationale"])

    def test_it_does_not_claim_a_model_confidence(self, store):
        """
        `extraction_confidence` is the model's self-assessment of its own
        reading. Writing "High" into a hand-typed row puts it at the top of
        every confidence-sorted view on the strength of nothing at all.
        """
        from stage2_extraction import manual_entry

        manual_entry.save_manual_claim(_claim(), curator="julia")
        row = store.load_table1_as_dataframe().iloc[-1]

        assert row["extraction_confidence"] not in ("High", "Medium", "Low")

    def test_the_qc_report_does_not_count_it_as_model_output(self, store):
        from stage2_extraction import manual_entry, qc_report

        before = qc_report.build_qc_report()
        manual_entry.save_manual_claim(
            _claim(), curator="julia", quotes=["Some sentence not in the text."]
        )
        after = qc_report.build_qc_report()

        assert after.n_kers == before.n_kers, (
            "a curator's row moved the model's row count"
        )
        assert after.verification_rate == before.verification_rate, (
            "a curator's unverifiable quote moved the model's verification rate"
        )
        assert after.n_curator_rows == 1, "and it is not silently dropped either"

    def test_an_assertion_with_no_source_is_stored_as_one(self, store):
        from stage2_extraction import manual_entry

        manual_entry.save_manual_claim(
            _claim(source_doi=""), curator="julia"
        )
        row = store.load_table1_as_dataframe().iloc[-1]

        assert row["source_doi"] == manual_entry.NO_SOURCE_DOI, (
            "a claim with no paper behind it must not acquire a plausible DOI"
        )


class TestCorrectingARow:

    def test_editing_an_extracted_row_marks_it_and_keeps_the_original(self, store):
        from stage2_extraction import manual_entry

        record_id = int(store.load_table1_as_dataframe().iloc[0]["record_id"])
        manual_entry.update_manual_claim(
            record_id,
            _claim(direction="negative", upstream_ke_name="Decreased sodium-channel activity",
                   upstream_ke_level="Molecular",
                   downstream_ke_name="Impaired oligodendrocyte differentiation",
                   downstream_ke_level="Cellular"),
            curator="julia",
            rationale="The paper reports an inverse relationship.",
        )

        row = store.load_record(record_id)
        assert row["origin"] == "curator_edited", (
            "a corrected extraction that still reads as 'llm' misattributes "
            "the curator's words to the model"
        )
        assert row["direction"] == "negative"

        history = store.load_record_history(record_id)
        assert len(history) == 1
        assert "positive" not in str(row["direction"])
        assert history.iloc[0]["curator"] == "julia"

    def test_a_no_op_edit_writes_no_history(self, store):
        """Saving a form without changing it is not a curation decision."""
        from stage2_extraction import manual_entry

        record_id = int(store.load_table1_as_dataframe().iloc[0]["record_id"])
        current = store.load_record(record_id)
        result = manual_entry.update_manual_claim(
            record_id,
            _claim(
                upstream_ke_name=current["upstream_ke_name"],
                upstream_ke_level=current["upstream_ke_level"],
                downstream_ke_name=current["downstream_ke_name"],
                downstream_ke_level=current["downstream_ke_level"],
                direction=current["direction"],
                relation_kind=current["relation_kind"],
                ker_adjacency=current["ker_adjacency"],
            ),
            curator="julia",
        )

        assert result["changed"] == []
        assert store.load_record_history(record_id).empty
        assert store.load_record(record_id)["origin"] == "llm"

    def test_deleting_archives_the_claim(self, store):
        record_id = int(store.load_table1_as_dataframe().iloc[0]["record_id"])
        store.delete_record(record_id, curator="julia", reason="misread figure")

        assert store.load_record(record_id) is None
        history = store.load_record_history(record_id)
        assert len(history) == 1
        assert history.iloc[0]["action"] == "deleted"
        assert history.iloc[0]["reason"] == "misread figure"

    def test_the_source_of_a_claim_cannot_be_edited(self, store):
        """
        Reattributing a claim to a paper that never made it is the one edit
        that turns a correction into a fabrication.
        """
        record_id = int(store.load_table1_as_dataframe().iloc[0]["record_id"])
        store.update_table1_row(
            record_id, {"source_doi": "10.9999/not-this-paper"}, curator="julia"
        )
        assert store.load_record(record_id)["source_doi"] == "10.1234/extracted"


class TestAssertedKeyEvents:

    def test_a_manual_key_event_survives_renormalization(self, store):
        """
        The bug this exists to prevent: `replace_canonical_kes` deletes the
        whole canonical table and rebuilds it from raw labels. An event a
        curator added because no paper named it cannot be rebuilt from raw
        labels — so a plain rebuild deletes precisely the events that were
        added for being unrebuildable.
        """
        from stage2_extraction import ke_normalizer

        canonical_id = store.create_manual_canonical_ke(
            "Auditory hypersensitivity", "Individual",
            curator="julia", rationale="The known adverse outcome.",
        )
        ke_normalizer.normalize_table1(
            store.load_table1_as_dataframe(), threshold=0.86, ols4_enabled=False
        )

        surviving = store.load_canonical_kes()
        assert canonical_id in set(surviving["canonical_id"]), (
            "re-running normalization deleted a curator-asserted Key Event"
        )
        row = surviving[surviving["canonical_id"] == canonical_id].iloc[0]
        assert row["origin"] == "curator"

    def test_a_derived_event_supersedes_the_placeholder(self, store):
        """
        Once the corpus names the event, the assertion has been overtaken.
        Keeping both would put the same Key Event on the map twice.
        """
        from stage2_extraction import ke_normalizer, manual_entry

        store.create_manual_canonical_ke(
            "Reduced myelination", "Tissue", curator="julia", rationale="AO",
        )
        manual_entry.save_manual_claim(_claim(), curator="julia")
        ke_normalizer.normalize_table1(
            store.load_table1_as_dataframe(), threshold=0.86, ols4_enabled=False
        )

        names = [
            str(n).casefold()
            for n in store.load_canonical_kes()["canonical_name"]
        ]
        assert names.count("reduced myelination") == 1

    def test_an_unknown_level_is_refused(self, store):
        with pytest.raises(ValueError):
            store.create_manual_canonical_ke("Something", "Subatomic")


class TestValidation:

    def test_a_relationship_needs_two_different_ends(self, store):
        from stage2_extraction import manual_entry

        problems = manual_entry.validate(
            _claim(downstream_ke_name="Impaired oligodendrocyte differentiation")
        )
        assert any("same event" in p for p in problems)

    def test_a_rationale_is_required(self, store):
        from stage2_extraction import manual_entry

        problems = manual_entry.validate(_claim(entry_rationale="  "))
        assert any("why" in p.lower() for p in problems)

    def test_every_problem_is_reported_at_once(self, store):
        """One message at a time turns one correction into five round trips."""
        from stage2_extraction import manual_entry

        problems = manual_entry.validate(
            {"upstream_ke_name": "", "downstream_ke_name": ""}
        )
        assert len(problems) > 3

    @pytest.mark.parametrize("level", list(KE_LEVEL_ORDER))
    def test_every_declared_level_is_accepted(self, store, level):
        from stage2_extraction import manual_entry

        problems = manual_entry.validate(
            _claim(upstream_ke_level=level, downstream_ke_level=level)
        )
        assert not any("level" in p for p in problems)


class TestQuoteVerification:

    def test_a_pasted_quote_is_located_in_stored_text(self, store):
        """
        The case that makes manual entry worth building rather than tolerating:
        a curator quoting the paper correctly should get the same verified flag
        the model would have got.
        """
        from stage2_extraction import manual_entry

        class Chunk:
            chunk_id = "c1"
            text = (
                "Myelin basic protein staining was reduced by 40% in the "
                "corpus callosum of treated animals at postnatal day 21."
            )
            section = "Results"
            section_kind = "results"
            page_start = 7
            page_end = 7
            char_start = 0
            char_end = 120
            relevance_score = 1.0
            selected = True

        store.store_chunks(
            [Chunk()], source_doi="10.1234/extracted", source_filename="paper.pdf"
        )

        located = manual_entry.verify_quote(
            "Myelin basic protein staining was reduced by 40% in the corpus "
            "callosum of treated animals",
            source_doi="10.1234/extracted",
        )
        assert located["searched"] is True
        assert located["verified"] is True
        assert located["page_start"] == 7

    def test_an_unstored_paper_is_not_reported_as_a_failed_check(self, store):
        """
        "Not checked" and "checked and not found" are different facts, and
        showing the first as the second accuses a curator of misquoting a
        paper the tool simply never read.
        """
        from stage2_extraction import manual_entry

        located = manual_entry.verify_quote(
            "Some sentence long enough to be searched for.",
            source_doi="10.5555/never-ingested",
        )
        assert located["searched"] is False
        assert located["verified"] is False
