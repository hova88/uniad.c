#!/usr/bin/env python3
import hashlib
commit = "609ee083ea51c3521c323f1279dfc4cee0e60467"
expected = "04dc11a82f7de530bd86fdac9bda0b4b65298e38d9732b74248ba84cf82b15a5"
actual = hashlib.sha256(commit.encode()).hexdigest()
assert actual == expected, (actual, expected)
print("provenance pin: ok")
