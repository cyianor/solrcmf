from sklearn.utils import get_tags

from solrcmf import LowRankImputation, SolrCMF, SolrCMFCV


def test_solrcmf_tags():
    """SolrCMF declares dict input with NaN support."""
    tags = get_tags(SolrCMF(structure_penalty=0.1, max_rank=2))

    assert tags.input_tags.dict
    assert not tags.input_tags.two_d_array
    assert tags.input_tags.allow_nan


def test_solrcmfcv_tags():
    """SolrCMFCV declares dict input with NaN support."""
    tags = get_tags(SolrCMFCV(structure_penalty=0.1, max_rank=2))

    assert tags.input_tags.dict
    assert not tags.input_tags.two_d_array
    assert tags.input_tags.allow_nan


def test_lowrankimputation_tags():
    """LowRankImputation declares NaN support."""
    tags = get_tags(LowRankImputation())

    assert not tags.input_tags.dict
    assert tags.input_tags.two_d_array
    assert tags.input_tags.allow_nan
