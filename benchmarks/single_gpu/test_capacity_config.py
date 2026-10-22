"""Large-case capacity configurations must preserve scientific settings."""
import pytest


def test_capacity_storage_settings():
    from capacity import configuration
    from comprehensive import settings
    for case in ['tamoxifen_tzvp_b3lyp','taxol_svp_b3lyp','valinomycin_svp_pbe','valinomycin_svp_b3lyp']:
        config=configuration(case)
        assert config['scientific']==settings(case)
        assert config['cache_bytes']==0
        assert config['buffer_bytes']==4*1024**3
        assert config['storage']==('disk' if 'valinomycin' in case else 'resident')
    with pytest.raises(ValueError): configuration('vitamin_svp_pbe')
