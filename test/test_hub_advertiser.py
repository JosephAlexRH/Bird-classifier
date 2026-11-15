"""The server answers for birdhub.local so cameras need no typed-in IP."""

from __future__ import annotations

import socket
import unittest

from server.hub_advertiser import (
    SERVICE_TYPE,
    HubAdvertiser,
    lan_ipv4_addresses,
    normalize_hub_name,
)


class FakeInfo:
    def __init__(self, type_, name, **kwargs):
        self.type, self.name, self.kwargs = type_, name, kwargs


class FakeZeroconf:
    def __init__(self):
        self.calls: list[tuple[str, FakeInfo]] = []
        self.closed = False

    def register_service(self, info):
        self.calls.append(("register", info))

    def update_service(self, info):
        self.calls.append(("update", info))

    def unregister_service(self, info):
        self.calls.append(("unregister", info))

    def close(self):
        self.closed = True


class HubAdvertiserTests(unittest.TestCase):
    def make(self, addresses, name="birdhub"):
        self.zeroconf = FakeZeroconf()
        self.lines: list[str] = []
        return HubAdvertiser(
            name,
            5000,
            address_source=lambda: list(addresses),
            zeroconf_factory=lambda: self.zeroconf,
            service_info_factory=FakeInfo,
            log=self.lines.append,
        )

    def test_names_are_normalized_and_validated(self) -> None:
        self.assertEqual(normalize_hub_name("BirdHub.local"), "birdhub")
        self.assertEqual(normalize_hub_name(" garden-hub "), "garden-hub")
        for bad in ("", "bird hub", "-hub", "hub_1", "a" * 64):
            with self.assertRaises(ValueError):
                normalize_hub_name(bad)

    def test_registers_the_host_name_with_this_machines_address(self) -> None:
        addresses = ["192.168.1.20"]
        advertiser = self.make(addresses)
        advertiser.refresh()
        action, info = self.zeroconf.calls[0]
        self.assertEqual(action, "register")
        self.assertEqual(info.type, SERVICE_TYPE)
        self.assertEqual(info.name, f"birdhub.{SERVICE_TYPE}")
        self.assertEqual(info.kwargs["server"], "birdhub.local.")
        self.assertEqual(info.kwargs["port"], 5000)
        self.assertEqual(info.kwargs["addresses"], [socket.inet_aton("192.168.1.20")])
        self.assertIn("birdhub.local", self.lines[0])

    def test_follows_an_address_change_and_stays_quiet_otherwise(self) -> None:
        addresses = ["192.168.1.20"]
        advertiser = self.make(addresses)
        advertiser.refresh()
        advertiser.refresh()
        self.assertEqual(len(self.zeroconf.calls), 1)
        addresses[:] = ["192.168.1.99"]
        advertiser.refresh()
        action, info = self.zeroconf.calls[-1]
        self.assertEqual(action, "update")
        self.assertEqual(info.kwargs["addresses"], [socket.inet_aton("192.168.1.99")])

    def test_no_network_withdraws_then_recovers(self) -> None:
        addresses: list[str] = []
        advertiser = self.make(addresses)
        advertiser.refresh()
        advertiser.refresh()
        self.assertEqual(self.zeroconf.calls, [])
        self.assertEqual(len(self.lines), 1)  # said once, not every check
        addresses[:] = ["10.42.0.1"]
        advertiser.refresh()
        self.assertEqual(self.zeroconf.calls[-1][0], "register")
        addresses[:] = []
        advertiser.refresh()
        self.assertEqual(self.zeroconf.calls[-1][0], "unregister")

    def test_stop_unregisters_and_closes(self) -> None:
        advertiser = self.make(["192.168.1.20"])
        advertiser.refresh()
        advertiser.stop()
        self.assertEqual(self.zeroconf.calls[-1][0], "unregister")
        self.assertTrue(self.zeroconf.closed)

    def test_a_specific_bind_address_is_advertised_as_is(self) -> None:
        self.assertEqual(lan_ipv4_addresses("192.168.1.20"), ["192.168.1.20"])
        self.assertEqual(lan_ipv4_addresses("127.0.0.1"), [])


if __name__ == "__main__":
    unittest.main()
