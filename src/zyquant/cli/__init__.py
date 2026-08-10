def main(argv=None):
    from .main import main as entrypoint
    return entrypoint(argv)


__all__ = ["main"]
