def run():
    from connectit.pieces import split_pieces, piece_hashes, verify_and_reassemble
    data = b"Hello World" * 100
    parts = split_pieces(data, 32)
    hashes = piece_hashes(parts)
    out = verify_and_reassemble(parts, hashes)
    assert out == data

