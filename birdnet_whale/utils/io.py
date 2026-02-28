from pathlib import Path
from typing import Dict, List, Iterable


AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


def collect_audio_files_by_class(audio_dir: str, exts: Iterable[str] = AUDIO_EXTS) -> Dict[str, List[Path]]:
    root = Path(audio_dir)
    if not root.exists():
        return {}
    class_dirs = [p for p in root.iterdir() if p.is_dir()]
    class_dirs = sorted(class_dirs, key=lambda p: p.name)

    file_map: Dict[str, List[Path]] = {}
    for cls in class_dirs:
        files = [p for p in cls.rglob("*") if p.suffix.lower() in exts]
        files = sorted(files, key=lambda p: p.name)
        file_map[cls.name] = files
    return file_map
