import os

# Settings require real secrets; tests only need placeholders
os.environ.setdefault("JWT_SECRET", "test-secret-" + "x" * 32)
os.environ.setdefault("ADMIN_PASSWORD", "test-password")
