"""Render freshly executed gallery files through MkDocs' generated-file API."""

from pathlib import Path
from tempfile import TemporaryDirectory

from build_gallery import build_gallery
from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File


def on_files(files, config):
    try:
        with TemporaryDirectory(prefix="vbpca-gallery-") as directory:
            output = Path(directory)
            build_gallery(Path(__file__).resolve().parents[1], output)
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    files.append(
                        File.generated(
                            config,
                            "gallery/" + path.relative_to(output).as_posix(),
                            content=path.read_bytes(),
                        )
                    )
    except Exception as error:
        detail = getattr(error, "stderr", "")
        raise PluginError(f"Executable gallery failed: {error}\n{detail}") from error
    return files
