from hydra.core.config_search_path import ConfigSearchPath
from hydra.plugins.search_path_plugin import SearchPathPlugin

__all__ = [
    "DistShiftPlugin",
]


class DistShiftPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        search_path.prepend(provider="ds", path="file://external_confs")
