import os, sys, xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from compression.section_parser import parse_paper_sections
from compression.fact_sheet import compress_paper

MINIMAL_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>'
    '<article><body>'
    '<sec><title>Introduction</title>'
    '<p>Ecological communities are shaped by abiotic factors across spatial scales.</p>'
    '</sec>'
    '<sec><title>Materials and Methods</title>'
    '<p>We sampled 120 plots. Species abundance was estimated using cover-abundance scores.</p>'
    '<sec><title>Data Collection</title>'
    '<p>Field surveys conducted June-August 2022. Shannon diversity indices calculated per plot.</p>'
    '</sec>'
    '</sec>'
    '<sec><title>Results</title>'
    '<p>Species richness peaked in montane zones (p less than 0.001, r2 = 0.62). PERMANOVA F = 18.7.</p>'
    '</sec>'
    '<sec><title>Discussion</title>'
    '<p>Findings confirm the mid-domain effect hypothesis. Results have implications for conservation.</p>'
    '</sec>'
    '<sec><title>Conclusion</title>'
    '<p>Elevation gradients drive significant variation in community composition and biodiversity.</p>'
    '</sec>'
    '</body></article>'
)


def _parse(xml_string):
    from agents.agent_research import _parse_pmc_xml_to_structured_text
    root = ET.fromstring(xml_string)
    return _parse_pmc_xml_to_structured_text(root)


def _paper(text, abstract=''):
    return {
        'title': 'PMC Regression Test',
        'authors': ['Author A'],
        'year': 2023,
        'url': 'https://europepmc.org/article/PMC99999',
        'abstract': abstract,
        'full_text': text,
        'coverage_type': 'full_paper',
    }


class TestPMCXMLParsing:
    def test_import(self):
        from agents.agent_research import _parse_pmc_xml_to_structured_text
        assert callable(_parse_pmc_xml_to_structured_text)

    def test_produces_multiline_text(self):
        text = _parse(MINIMAL_XML)
        assert text, 'Parser returned empty text.'
        assert '\n' in text, 'Parser flattened to single line -- section boundaries destroyed.'

    def test_section_titles_present_as_lines(self):
        text = _parse(MINIMAL_XML)
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        found = sum(1 for l in lines if any(h in l for h in ['Introduction', 'Methods', 'Results', 'Discussion']))
        assert found >= 2, f'Expected >= 2 heading lines, got {found}. Lines: {lines}'

    def test_paragraph_content_preserved(self):
        text = _parse(MINIMAL_XML)
        assert 'ecological communities' in text.lower() or 'Ecological communities' in text

    def test_nested_section_content_preserved(self):
        text = _parse(MINIMAL_XML)
        assert 'Shannon diversity' in text or 'Field surveys' in text

    def test_no_raw_xml_tags(self):
        import re
        text = _parse(MINIMAL_XML)
        tags = re.findall(r'<[a-zA-Z/][^>]*>', text)
        assert not tags, f'Raw XML tags in output: {tags[:5]}'


class TestPMCSectionDetection:
    def test_not_only_unknown_abstract(self):
        text = _parse(MINIMAL_XML)
        sections = parse_paper_sections(text)
        meaningful = set(sections.keys()) - {'unknown'}
        assert len(meaningful) >= 2, (
            f'Root bug present: only detected {set(sections.keys())}. '
            'PMC structure still being destroyed before section parsing.'
        )

    def test_introduction_detected(self):
        text = _parse(MINIMAL_XML)
        sections = parse_paper_sections(text)
        assert 'introduction' in sections, f'introduction not in {list(sections.keys())}'

    def test_methodology_detected(self):
        text = _parse(MINIMAL_XML)
        sections = parse_paper_sections(text)
        assert 'methodology' in sections, f'methodology not in {list(sections.keys())}'

    def test_results_detected(self):
        text = _parse(MINIMAL_XML)
        sections = parse_paper_sections(text)
        assert 'results' in sections, f'results not in {list(sections.keys())}'

    def test_discussion_detected(self):
        text = _parse(MINIMAL_XML)
        sections = parse_paper_sections(text)
        assert 'discussion' in sections, f'discussion not in {list(sections.keys())}'


class TestPMCCompressionQuality:
    def test_sections_not_only_unknown_abstract(self):
        text = _parse(MINIMAL_XML)
        paper = _paper(text)
        fs = compress_paper(paper, query='ecology environment', paper_id='PMC-REG-001')
        assert fs is not None
        only_bad = set(fs.sections.keys()) <= {'unknown', 'abstract'}
        assert not only_bad, f'Still producing only {set(fs.sections.keys())}. Root bug not fixed.'

    def test_compression_ratio_below_97(self):
        text = _parse(MINIMAL_XML)
        paper = _paper(text)
        fs = compress_paper(paper, query='ecology', paper_id='PMC-REG-002')
        if fs.metrics.original_token_count > 0:
            ratio = fs.metrics.compression_ratio
            assert ratio < 0.97, f'Compression ratio {ratio*100:.1f}% -- still near root-bug 97.9%.'

    def test_technical_sentences_nonzero(self):
        text = _parse(MINIMAL_XML)
        paper = _paper(text)
        fs = compress_paper(paper, query='ecology species richness', paper_id='PMC-REG-003')
        total = sum(s.technical_sentence_count for s in fs.sections.values())
        assert total > 0, (
            'Technical sentences=0 for paper with p<0.001, r2=0.62, PERMANOVA, Shannon diversity. '
            'Technical preservation patterns may not cover ecology vocabulary.'
        )


class TestHasRealAbstract:
    def test_import(self):
        from agents.agent_research import has_real_abstract
        assert callable(has_real_abstract)

    def test_empty_rejected(self):
        from agents.agent_research import has_real_abstract
        assert not has_real_abstract('')
        assert not has_real_abstract('   ')

    def test_placeholders_rejected(self):
        from agents.agent_research import has_real_abstract
        placeholders = [
            'No abstract available.',
            'No abstract available',
            'Abstract not available',
            'Not available',
            'N/A',
            'Abstract metadata fetched from OpenAlex Open DOI index.',
        ]
        for p in placeholders:
            assert not has_real_abstract(p), f'Placeholder not rejected: {p!r}'

    def test_short_string_rejected(self):
        from agents.agent_research import has_real_abstract
        assert not has_real_abstract('This paper presents a method.')

    def test_genuine_abstract_accepted(self):
        from agents.agent_research import has_real_abstract
        text = (
            'We propose a novel transformer architecture for clinical text classification. '
            'Our model achieves 94.7 percent accuracy on the MIMIC-III benchmark using 110M parameters. '
            'The system demonstrates 37 percent reduction in inference latency versus BERT-large.'
        )
        assert has_real_abstract(text)

    def test_ecology_abstract_accepted(self):
        from agents.agent_research import has_real_abstract
        text = (
            'Understanding species richness across elevation gradients is fundamental to ecology. '
            'We sampled 120 plots and found biodiversity peaked at intermediate elevations, '
            'supporting the mid-domain effect hypothesis with Shannon diversity correlating with precipitation.'
        )
        assert has_real_abstract(text)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
