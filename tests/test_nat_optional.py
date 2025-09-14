import asyncio


def run():
    from connectit.nat import try_upnp_map, try_stun
    ok, ip = try_upnp_map(5555)
    assert ok in (True, False)
    assert (ip is None) or isinstance(ip, str)

    async def _s():
        out = await try_stun()
        assert (out is None) or isinstance(out, str)
    asyncio.run(_s())

