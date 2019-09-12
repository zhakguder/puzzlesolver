from pathlib import Path


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent

def divider(by):
    def divider_():
        x = 1
        while x>=0.05:
            try:
                x /= by
                yield x
            except:
                break
        return
    return divider_

halver = divider(2)
