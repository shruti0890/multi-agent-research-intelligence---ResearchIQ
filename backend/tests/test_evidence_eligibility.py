import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest

from agents.agent_gap import is_evidence_eligible, cluster_and_analyze_gaps
from agents.agent_research import _detect_domain_mismatch
from compression.section_parser import NON_RESEARCH_SECTIONS, parse_paper_sections
from compression.fact_sheet import compress_paper
from schemas import Agent1ResearchOutput, PaperMetadata


class TestEvidenceEligibility:
    def test_eligible_coverage_types(self):
        class MockPaper:
            def __init__(self, cov, access=None):
                self.coverage_type = cov
                if access:
                    self.access_status = access

        assert is_evidence_eligible(MockPaper('full_paper')) is True
        assert is_evidence_eligible(MockPaper('partial_paper')) is True
        assert is_evidence_eligible(MockPaper('full_paper', access='FULL_TEXT_AVAILABLE')) is True

    def test_ineligible_coverage_types(self):
        class MockPaper:
            def __init__(self, cov, access=None):
                self.coverage_type = cov
                if access:
                    self.access_status = access

        assert is_evidence_eligible(MockPaper('abstract_only')) is False
        assert is_evidence_eligible(MockPaper('abstract_only', access='ABSTRACT_ONLY')) is False
        assert is_evidence_eligible(MockPaper('unavailable')) is False
        assert is_evidence_eligible(MockPaper('unavailable', access='UNAVAILABLE')) is False
        assert is_evidence_eligible(MockPaper('none')) is False
        assert is_evidence_eligible(MockPaper('')) is False


class TestNonResearchSectionSkipping:
    def test_non_research_sections_constant(self):
        assert 'references' in NON_RESEARCH_SECTIONS
        assert 'acknowledgments' in NON_RESEARCH_SECTIONS
        assert 'supplementary_material' in NON_RESEARCH_SECTIONS

    def test_non_research_sections_skipped_in_fact_sheet(self):
        text = (
            'Introduction\n\n'
            'We study ecological biodiversity in temperate forests across elevation.\n\n'
            'Methods\n\n'
            'We sampled 100 quadrats and measured species richness.\n\n'
            'References\n\n'
            '1. Smith et al. 2020. Journal of Ecology. 2. Jones et al. 2021. Nature.\n'
        )
        paper_dict = {
            'title': 'Test Ecology Paper',
            'authors': ['Author'],
            'year': 2023,
            'url': 'http://example.com',
            'abstract': 'We study ecological biodiversity in temperate forests.',
            'full_text': text,
            'coverage_type': 'full_paper',
        }
        fs = compress_paper(paper_dict, query='ecology biodiversity', paper_id='P001')
        assert 'references' not in fs.sections
        assert 'introduction' in fs.sections or 'methodology' in fs.sections


class TestDomainMismatchDetector:
    def test_ev_ecosystem_penalized_for_ecology_query(self):
        penalty = _detect_domain_mismatch(
            query='Ecology and ecosystem',
            title='Development of an Integrated EV Service Ecosystem',
            abstract='This paper proposes a framework for electric vehicle charging service ecosystems in smart cities.'
        )
        assert penalty < 0.6, f'Expected penalty for EV ecosystem under ecology query, got {penalty}'

    def test_software_ecosystem_penalized_for_ecology_query(self):
        penalty = _detect_domain_mismatch(
            query='Ecosystem and environment',
            title='Analyzing Developer Networks in the Open Source Software Ecosystem',
            abstract='We examine package dependencies in npm and github software ecosystems.'
        )
        assert penalty < 0.6, f'Expected penalty for software ecosystem under ecology query, got {penalty}'

    def test_genuine_ecological_ecosystem_not_penalized(self):
        penalty = _detect_domain_mismatch(
            query='Ecology and ecosystem',
            title='Forest Ecosystem Resilience and Biodiversity Under Climate Change',
            abstract='We assess species richness, biomass, and trophic interactions in temperate forest ecosystems.'
        )
        assert penalty == 1.0, f'Expected 1.0 for genuine ecology paper, got {penalty}'

    def test_non_ecology_query_not_penalized(self):
        penalty = _detect_domain_mismatch(
            query='EV Service Architecture',
            title='Development of an Integrated EV Service Ecosystem',
            abstract='This paper proposes a framework for electric vehicle charging service ecosystems.'
        )
        assert penalty == 1.0, f'Expected 1.0 when query is not ecology-focused, got {penalty}'


class TestUnavailablePaperGapAnalysis:
    def test_unavailable_papers_receive_static_marker_and_not_fabricated_claims(self):
        papers = [
            PaperMetadata(
                title='Unavailable Paper 1',
                authors=['Author A'],
                year=2024,
                abstract='',
                relevance_rank='Low',
                notebook_summary='',
                technical_execution='',
                datasets=[],
                url='',
                relevance_score=0.5,
                coverage_type='unavailable',
                paper_id='P001',
            ),
            PaperMetadata(
                title='Unavailable Paper 2',
                authors=['Author B'],
                year=2024,
                abstract='',
                relevance_rank='Low',
                notebook_summary='',
                technical_execution='',
                datasets=[],
                url='',
                relevance_score=0.5,
                coverage_type='unavailable',
                paper_id='P002',
            )
        ]
        research_output = Agent1ResearchOutput(query='Ecology and ecosystem', papers=papers)
        gap_output = cluster_and_analyze_gaps(research_output)

        assert len(gap_output.gaps) == 2
        for g in gap_output.gaps:
            assert 'Evidence unavailable' in g.description
            assert g.severity == 'Low'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
