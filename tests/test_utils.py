def run():
    from connectit.utils import new_id, sha256_hex, hash_password, gen_salt
    a = new_id("x")
    b = new_id("x")
    assert a != b and a.startswith("x-"), "new_id uniqueness/prefix"
    assert sha256_hex("abc") == sha256_hex("abc"), "sha256 deterministic"
    salt = gen_salt()
    assert hash_password("p", salt) == hash_password("p", salt), "hash password stable"

