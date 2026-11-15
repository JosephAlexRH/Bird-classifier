"""Advertise this server on the local network as ``<name>.local``.

The cameras look the hub up by name (``server_host`` in their config.json,
``birdhub.local`` by default), so nobody has to type the hub's IP address into
a camera. This module answers that lookup from the server process itself using
mDNS, which works the same on a Windows laptop and on a Raspberry Pi and does
not require renaming the machine.

``zeroconf`` is an optional dependency: without it the server still runs and
prints how to enable the name (cameras then fall back to ``server_ip``).
"""

from __future__ import annotations

import re
import socket
import threading
from typing import Any, Callable

SERVICE_TYPE = "_birdhub._tcp.local."
DEFAULT_HUB_NAME = "birdhub"
ADDRESS_CHECK_SECONDS = 30.0
_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def normalize_hub_name(value: str) -> str:
    """Return a valid mDNS host label from ``birdhub`` or ``birdhub.local``."""
    name = (value or "").strip().lower().rstrip(".")
    if name.endswith(".local"):
        name = name[: -len(".local")]
    if not _LABEL.match(name):
        raise ValueError(
            "Hub name must be 1-63 letters, digits or hyphens (for example: birdhub)."
        )
    return name


def lan_ipv4_addresses(bind_host: str | None = None) -> list[str]:
    """IPv4 addresses the cameras can reach this machine on.

    A specific ``--host`` wins. Otherwise the address of the default route is
    used, which avoids advertising VPN or virtual-machine adapters. With no
    default route (for example a Pi acting as its own hotspot) every
    non-loopback adapter address is offered.
    """
    if bind_host and bind_host not in {"0.0.0.0", "::", ""}:
        try:
            socket.inet_aton(bind_host)
        except OSError:
            pass
        else:
            return [] if bind_host.startswith("127.") else [bind_host]
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # UDP connect sends nothing; it only picks a route
            address = probe.getsockname()[0]
        if address and not address.startswith(("127.", "0.")):
            return [address]
    except OSError:
        pass
    try:
        import ifaddr
    except ImportError:
        return []
    found: list[str] = []
    for adapter in ifaddr.get_adapters():
        for ip in adapter.ips:
            if isinstance(ip.ip, str) and not ip.ip.startswith(("127.", "169.254.")) and ip.ip not in found:
                found.append(ip.ip)
    return found


class HubAdvertiser:
    """Keeps ``<name>.local`` pointing at this machine until stopped."""

    def __init__(
        self,
        name: str,
        port: int,
        *,
        address_source: Callable[[], list[str]],
        zeroconf_factory: Callable[[], Any],
        service_info_factory: Callable[..., Any],
        log: Callable[[str], None] | None = None,
        check_seconds: float = ADDRESS_CHECK_SECONDS,
    ) -> None:
        self.name = normalize_hub_name(name)
        self.port = int(port)
        self._address_source = address_source
        self._zeroconf_factory = zeroconf_factory
        self._service_info_factory = service_info_factory
        self._log = log or (lambda message: print(message, flush=True))
        self._check_seconds = check_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._zeroconf: Any = None
        self._info: Any = None
        self.addresses: list[str] = []
        self._waiting_for_address = False

    @property
    def hostname(self) -> str:
        return f"{self.name}.local"

    def _build_info(self, addresses: list[str]) -> Any:
        return self._service_info_factory(
            SERVICE_TYPE,
            f"{self.name}.{SERVICE_TYPE}",
            port=self.port,
            addresses=[socket.inet_aton(address) for address in addresses],
            server=f"{self.hostname}.",
            properties={"path": "/"},
        )

    def refresh(self) -> None:
        """Register, or re-register after this machine's address changed."""
        addresses = self._address_source()
        if not addresses:
            if not self._waiting_for_address:
                self._log(f"Hub name: no network address yet; {self.hostname} is not being advertised.")
            self._waiting_for_address = True
            self._withdraw()
            return
        self._waiting_for_address = False
        if addresses == self.addresses and self._info is not None:
            return
        if self._zeroconf is None:
            self._zeroconf = self._zeroconf_factory()
        info = self._build_info(addresses)
        if self._info is None:
            self._zeroconf.register_service(info)
        else:
            self._zeroconf.update_service(info)
        self._info, self.addresses = info, addresses
        self._log(
            f"Hub name: cameras can find this server as {self.hostname} "
            f"({', '.join(addresses)}, port {self.port})."
        )

    def _withdraw(self) -> None:
        if self._zeroconf is not None and self._info is not None:
            try:
                self._zeroconf.unregister_service(self._info)
            except Exception:  # noqa: BLE001 - shutting down regardless
                pass
        self._info, self.addresses = None, []

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.refresh()
            except Exception as error:  # noqa: BLE001 - advertising must never stop the server
                if type(error).__name__ == "NonUniqueNameException":
                    self._log(
                        f"Hub name: another server on this network is already using {self.hostname}. "
                        "Stop it, or start this one with a different --hub-name."
                    )
                    return
                self._log(f"Hub name: could not advertise {self.hostname}: {error}")
            self._stop.wait(self._check_seconds)

    def start(self) -> "HubAdvertiser":
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="hub-advertiser", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._withdraw()
        if self._zeroconf is not None:
            try:
                self._zeroconf.close()
            except Exception:  # noqa: BLE001
                pass
            self._zeroconf = None


def start_hub_advertiser(
    name: str,
    port: int,
    *,
    bind_host: str | None = None,
    log: Callable[[str], None] | None = None,
) -> HubAdvertiser | None:
    """Start advertising in the background; returns None when it cannot."""
    log = log or (lambda message: print(message, flush=True))
    try:
        from zeroconf import ServiceInfo, Zeroconf
    except ImportError:
        log(
            "Hub name: the 'zeroconf' package is not installed, so cameras cannot find this "
            "server by name and will use their server_ip setting. Run: pip install zeroconf"
        )
        return None
    try:
        advertiser = HubAdvertiser(
            name,
            port,
            address_source=lambda: lan_ipv4_addresses(bind_host),
            zeroconf_factory=Zeroconf,
            service_info_factory=ServiceInfo,
            log=log,
        )
    except ValueError as error:
        log(f"Hub name: {error}")
        return None
    return advertiser.start()
