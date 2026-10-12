from app.citation_mapper import _match_citation, _fallback_reference, split_sentences


def test_author_year_match():
    ref = _fallback_reference("Pradhan R, Lahiri A (2022) Explainable AI: foundations and applications. IEEE", 1)
    sentence = "In addressing firms' incentives to conceal AI explanations, the prevailing regulatory approach often mandates full explanations to end users (Pradhan et al. 2022)."
    ok, method = _match_citation(sentence, ref)
    assert ok
    assert "2022" in method


def test_latex_cite_key_match():
    ref = _fallback_reference("Explainable AI: Foundations", 1, bib_key="pradhan2022xai")
    ok, _ = _match_citation("This result is widely discussed [CITE:pradhan2022xai].", ref)
    assert ok


def test_sentence_split():
    parts = split_sentences("We study the claim and cite the prior work (Smith et al. 2022). Another sufficiently long sentence follows.")
    assert len(parts) >= 2
