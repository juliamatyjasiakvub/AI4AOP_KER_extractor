"""
A Key Event is a change. A bare noun phrase is not one.

The corpus this guards against is real. Twelve of twenty-eight extracted rows
came back naming the upstream event "Voltage-gated sodium channels" — a thing,
not an event — and that single node then absorbed:

    * seven rows reporting the quantity DECREASED and five reporting it INCREASED
    * patch-clamp current density, Nav1.6 immunolabeling, and RNA-seq transcript
      counts, which are function, protein and message and are three Key Events
    * ten different cell types and compartments

None of that is visible from the name, so the node reads as one well-evidenced
Key Event supported by many papers. It is not; it is several events
stacked on top of each other, and the confidence score rewards the stack.

The prompt now forbids bare entities. Prompts are not guarantees — a rule a
model honours on most papers is one it can drop on the next, silently
— so `name_problems` re-checks every name the model invents and attaches a
warning to the paper that produced it.

The second half of this file guards the classifier the check depends on. Its
first version flagged "altered Nav channel clustering at the heminode" and
"shortened oligodendroglial membrane protrusions/internodes", both of which
plainly state a change. A checker that cries wolf on correct names is worse
than no checker: it teaches the curator to ignore the warnings that matter.
"""

import pytest

from stage2_extraction.ker_extractor import name_problems
from stage2_extraction.semantic_merge import ObjectType, object_type


class TestBareEntitiesAreCaught:

    @pytest.mark.parametrize("label", [
        "Voltage-gated sodium channels",
        "Synaptic input from neurons",
        "Mitochondria",
        "Myelin basic protein",
    ])
    def test_a_thing_is_not_a_key_event(self, label):
        assert object_type(label) is ObjectType.ENTITY
        problems = name_problems(label)
        assert problems, f"{label!r} should have been flagged"
        assert "names a thing rather than a change" in problems[0]

    def test_the_warning_explains_the_consequence_not_just_the_rule(self):
        """
        "Not a Key Event" is a verdict a curator can disagree with. "Every
        paper reporting this rising and every paper reporting it falling will
        land on the same node" is a consequence they can check.
        """
        message = name_problems("Voltage-gated sodium channels")[0]
        assert "rising" in message and "falling" in message
        assert "transcript" in message

    def test_no_auto_suggested_rewrite(self):
        """
        An earlier version appended a suggested rename built by prefixing
        "decreased", which produced "decreased shortened internodes". A wrong
        suggestion beside a correct diagnosis discredits the diagnosis.
        """
        message = name_problems("Voltage-gated sodium channels")[0]
        assert "decreased voltage-gated" not in message.lower()

    def test_a_null_result_is_not_a_key_event_either(self):
        problems = name_problems("No change in OPC proliferation")
        assert problems and "did not change" in problems[0]

    def test_several_names_give_several_problems(self):
        assert len(name_problems("Mitochondria", "Voltage-gated sodium channels")) == 2

    def test_blank_names_are_ignored(self):
        assert name_problems(None, "", "   ") == []


class TestLegitimateNamesArePassed:
    """
    Every one of these is a real label from the corpus. A false positive here
    is the failure mode that makes the whole check worthless.
    """

    @pytest.mark.parametrize("label", [
        "decreased myelin basic protein expression",
        "increased oligodendrocyte progenitor cell proliferation",
        "decreased myelin thickness",
        "decreased presynaptic excitability",
        "loss of action potential firing in pre-oligodendrocytes",
        "restoration of compound action potential conduction",
        "impaired synaptic transmission and short-term plasticity",
        "conduction failure and reduced spike fidelity",
        "retinal ganglion cell death",
        "Oligodendrocyte differentiation",
        "Myelination of axons",
        "Acquisition of spiking properties",
        "auditory hypersensitivity",
    ])
    def test_a_stated_change_passes(self, label):
        assert object_type(label) is ObjectType.EVENT
        assert name_problems(label) == [], f"{label!r} was wrongly flagged"

    @pytest.mark.parametrize("label", [
        "altered Nav channel clustering at the heminode",
        "shortened oligodendroglial membrane protrusions/internodes",
    ])
    def test_an_unsigned_change_is_still_a_change(self, label):
        """
        The regression. "Altered" and "shortened" say something happened
        without saying which way on a signed axis, and neither matched any
        polarity pattern, so both were classified ENTITY alongside
        "voltage-gated sodium channels".
        """
        assert object_type(label) is ObjectType.EVENT
        assert name_problems(label) == []


class TestUnsignedChangesKeepNoPolarity:
    """
    Recognising "altered X" as an event must not give it a direction. The
    polarity guard refuses to merge labels whose directions conflict, and
    inventing a direction nobody stated would make it refuse merges on
    evidence that does not exist.
    """

    def test_altered_has_no_polarity(self):
        from stage2_extraction.semantic_merge import read_state

        state = read_state("altered Nav channel clustering at the heminode")
        assert not state.increase
        assert not state.decrease
        assert not state.restoration

    def test_signed_labels_still_read_their_direction(self):
        from stage2_extraction.semantic_merge import read_state

        assert read_state("decreased myelin thickness").decrease
        assert read_state("increased ROS").increase


class TestThePromptStatesTheRule:
    """
    The check is a safety net; the prompt is the fix. If the rule leaves the
    prompt, every run depends on the net.
    """

    def test_the_naming_rules_forbid_bare_entities(self):
        from stage2_extraction import ker_extractor

        persona = ker_extractor._PERSONA
        assert "A bare entity is NEVER a Key Event" in persona
        assert "Voltage-gated sodium channels" in persona

    def test_the_naming_rules_separate_function_protein_and_transcript(self):
        from stage2_extraction import ker_extractor

        persona = ker_extractor._PERSONA
        assert "three events, not one" in persona
        for word in ("current density", "protein", "transcript"):
            assert word in persona

    def test_the_pair_listing_task_repeats_the_rule(self):
        from stage2_extraction import ker_extractor

        task = ker_extractor._task_list_pairs()
        assert "not a Key Event name" in task


class TestThePromptStatesWhatTheTaskIs:
    """
    The prompt used to open on a persona line and go straight into thirty
    thousand characters about a neurotoxin, with a schema demanding mechanism
    of action and dose-response, and nothing anywhere saying where the text
    came from or what the output was for.

    That framing belongs there on its own merits — the naming and quotation
    rules only make sense once the output is known to be an evidence table —
    and every sentence of it is true of every run. It is not written to
    produce a reaction; it is written because it was missing.
    """

    def test_the_context_comes_first(self):
        from stage2_extraction import ker_extractor

        assert ker_extractor._PERSONA.startswith("CONTEXT FOR THIS TASK")

    def test_it_states_the_source_the_purpose_and_the_output(self):
        from stage2_extraction import ker_extractor

        persona = ker_extractor._PERSONA
        assert "peer-reviewed, published scientific paper" in persona
        assert "Adverse Outcome Pathway" in persona
        assert "bibliographic, not advisory" in persona
        assert "verbatim quotation" in persona

    def test_it_says_what_the_task_is_not(self):
        from stage2_extraction import ker_extractor

        persona = ker_extractor._PERSONA
        assert "not designing an experiment" in persona
        assert "not recommending an exposure" in persona

    def test_the_paper_still_follows_the_context(self):
        """
        The whole prefix is prompt-cached, so the ordering has to survive.
        """
        from stage2_extraction import ker_extractor

        prefix = ker_extractor._build_cached_prefix("SOME PAPER TEXT")
        assert prefix.startswith("CONTEXT FOR THIS TASK")
        assert prefix.rstrip().endswith("SOME PAPER TEXT")

    def test_screening_carries_the_same_framing(self):
        from schemas import PubMedRecord
        from stage1_search import screening

        record = PubMedRecord(
            pmid="1", doi="10.1/x", first_author="A", journal="J", year=2020,
            title="T", abstract="A.", query_used="q",
        )
        prompt = screening._build_prompt(record, "query", None, None)
        assert prompt.lstrip().startswith("CONTEXT FOR THIS TASK")
        # The screening prompt is a wrapped f-string, so phrases straddle
        # newlines. Collapse whitespace before matching rather than asserting
        # on an accident of where the line happened to break.
        flat = " ".join(prompt.split())
        assert "peer-reviewed" in flat
        assert "not advising on any substance" in flat
        assert "Adverse Outcome Pathway" in flat
