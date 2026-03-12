from importlib import import_module
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType
from typing import Any


def iter_plugin_entry_points() -> list[EntryPoint]:
    return list(entry_points(group="hubai.plugins"))


def load_cli_plugins() -> list[Any]:
    return [ep.load() for ep in iter_plugin_entry_points()]


def load_client_plugins() -> dict[str, Any]:
    plugins: dict[str, Any] = {}

    for ep in iter_plugin_entry_points():
        plugin = ep.load()

        # for HubAIClient we should expose the plugin module not the Cyclopts command object.
        if ep.attr == "app":
            plugin = import_module(ep.module)

        plugins[ep.name] = plugin

    return plugins
