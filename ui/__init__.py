"""
One module per step of the curation workflow.

The order of the modules is the order of the work:

    extract   → what the papers said, verbatim
    curate    → what those statements are, decided by a human
    approve   → sign-off, which is what unlocks the next step
    synthesis → the weight of evidence, per canonical KER
    aop_map   → the approved graph

Nothing in a later module reads from a stage that has not been approved. That
constraint is enforced in `stage2_extraction.workflow_state`, not here — the UI
only has to show why a section is locked.
"""
