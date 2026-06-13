"""Configuration package for AppleVision AI."""

from app.config.settings import DevelopmentConfig, ProductionConfig, TestingConfig

config_by_name: dict[str, type] = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}

__all__ = [
    "config_by_name",
    "DevelopmentConfig",
    "ProductionConfig",
    "TestingConfig",
]
