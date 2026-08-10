from __future__ import annotations

from importlib.metadata import entry_points
from dataclasses import dataclass
from typing import Any, Mapping

from .exceptions import PluginError
from .loading import load_object
from .versioning import FRAMEWORK_VERSION, PLUGIN_PROTOCOL_VERSION


@dataclass(frozen=True)
class PluginMetadata:
    name: str
    version: str
    kind: str
    protocol_version: str = PLUGIN_PROTOCOL_VERSION
    minimum_framework_version: str = "1.0.0"
    input_schema: str | None = None
    output_schema: str | None = None
    optional_dependencies: tuple[str, ...] = ()
    deterministic: bool = True


class PluginRegistry:
    """Small registry used by strategy projects without imposing inheritance."""

    def __init__(self):
        self._plugins: dict[str, dict[str, tuple[PluginMetadata, Any]]] = {}

    def register(
        self,
        kind: str,
        name: str,
        plugin: Any,
        metadata: PluginMetadata | Mapping[str, Any] | None = None,
    ) -> None:
        values = self._plugins.setdefault(kind, {})
        if name in values:
            raise PluginError(f"plugin already registered: {kind}:{name}")
        raw = metadata if metadata is not None else getattr(plugin, "plugin_metadata", None)
        if raw is None:
            raise PluginError(f"plugin metadata is required: {kind}:{name}")
        info = raw if isinstance(raw, PluginMetadata) else PluginMetadata(**raw)
        if info.name != name or info.kind != kind:
            raise PluginError(f"plugin metadata identity mismatch: {kind}:{name}")
        if info.protocol_version != PLUGIN_PROTOCOL_VERSION:
            raise PluginError(
                f"plugin {kind}:{name} protocol {info.protocol_version} is incompatible "
                f"with {PLUGIN_PROTOCOL_VERSION}"
            )
        if tuple(map(int, info.minimum_framework_version.split(".")[:2])) > tuple(
            map(int, FRAMEWORK_VERSION.split(".")[:2])
        ):
            raise PluginError(f"plugin {kind}:{name} requires a newer ZyQuant")
        values[name] = (info, plugin)

    def get(self, kind: str, name: str) -> Any:
        try:
            return self._plugins[kind][name][1]
        except KeyError as exc:
            raise PluginError(f"unknown plugin: {kind}:{name}") from exc

    def metadata(self, kind: str, name: str) -> PluginMetadata:
        try:
            return self._plugins[kind][name][0]
        except KeyError as exc:
            raise PluginError(f"unknown plugin: {kind}:{name}") from exc

    def discover(
        self,
        group_prefix: str = "zyquant",
        kinds: tuple[str, ...] | None = None,
    ) -> None:
        selected = kinds or (
            "data", "factors", "strategies", "models", "optimizers",
            "objectives", "reports", "execution_models", "cost_models",
            "capital_allocators",
        )
        for kind in selected:
            group = f"{group_prefix}.{kind}"
            for item in entry_points(group=group):
                if item.name not in self._plugins.get(kind, {}):
                    self.register(kind, item.name, item.load())

    def names(self, kind: str) -> tuple[str, ...]:
        return tuple(sorted(self._plugins.get(kind, {})))

    def resolve(
        self,
        kind: str,
        reference: str,
        project_root=None,
    ) -> Any:
        """Resolve an installed plugin name or a local ``module:object``."""
        if ":" in reference:
            plugin = load_object(reference, project_root)
            metadata = getattr(plugin, "plugin_metadata", None)
            if metadata is not None:
                info = metadata if isinstance(metadata, PluginMetadata) else PluginMetadata(**metadata)
                if info.kind != kind:
                    raise PluginError(
                        f"plugin kind mismatch: expected {kind}, got {info.kind}"
                    )
            return plugin
        if kind == "data":
            from zyquant.connectors import BUILTIN_DATA_SOURCES, builtin_factory
            if reference in BUILTIN_DATA_SOURCES:
                return builtin_factory(reference)
        return self.get(kind, reference)


plugins = PluginRegistry()
