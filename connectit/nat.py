from __future__ import annotations
from typing import Optional, Tuple


def try_upnp_map(port: int, proto: str = "TCP") -> Tuple[bool, Optional[str]]:
    try:
        import miniupnpc  # type: ignore
    except Exception:
        return False, None
    try:
        upnp = miniupnpc.UPnP()
        upnp.discoverdelay = 200
        upnp.discover()
        upnp.selectigd()
        external_ip = upnp.externalipaddress()
        upnp.addportmapping(port, proto, upnp.lanaddr, port, "ConnectIT", "")
        return True, external_ip
    except Exception:
        return False, None


async def try_stun() -> Optional[str]:
    try:
        from aiortc.contrib.media import MediaBlackhole  # type: ignore
        from aiortc import RTCIceGatherer
    except Exception:
        return None
    try:
        # Minimal STUN via ICE gatherer (requires network access and stun servers)
        gatherer = RTCIceGatherer()
        await gatherer.gather()
        for cand in gatherer.getLocalCandidates():
            if cand and cand.ip and cand.type == "srflx":
                return cand.ip
    except Exception:
        return None
    return None

