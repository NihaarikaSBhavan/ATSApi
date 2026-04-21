from atsv4.config import ATSConfig


def test_config_validate_accepts_default_weights() -> None:
    config = ATSConfig()
    config.validate()
