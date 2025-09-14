import asyncio


def run():
    from connectit.dht import DHTNode, announce_piece, find_providers

    async def main():
        dht = DHTNode()
        await dht.start()
        await announce_piece(dht, "hash123", "ws://127.0.0.1:4001")
        providers = await find_providers(dht, "hash123")
        assert "ws://127.0.0.1:4001" in providers

    asyncio.run(main())

