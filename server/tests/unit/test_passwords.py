from app.auth.passwords import hash_password, verify_password


def test_hash_and_verify() -> None:
    hashed = hash_password("secret123")
    assert verify_password("secret123", hashed)


def test_wrong_password_fails() -> None:
    hashed = hash_password("secret123")
    assert not verify_password("wrong", hashed)
