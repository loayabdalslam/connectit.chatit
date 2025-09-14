def run():
    from connectit.p2p import generate_join_link, parse_join_link
    link = generate_join_link("llmnet", "demo", "abc123", ["/ip4/127.0.0.1/tcp/4001/p2p/QmPeerID"]) 
    info = parse_join_link(link)
    assert info["network"] == "llmnet"
    assert info["model"] == "demo"
    assert info["hash"] == "abc123"
    assert len(info["bootstrap"]) == 1

