"""Build the cached Seamless transformer for reference best-fit fitting.

Caller code must initialize ``seamless_config`` before using this factory.
The factory never calls ``seamless_config.init()`` or resets configuration;
execution and result materialization also remain with the caller.
"""
from __future__ import annotations

import json
from pathlib import Path
import shlex
from collections.abc import Sequence

from seamless import Checksum
from seamless_transformer import Transformer
from seamless_transformer.cmd.file_load import files_to_checksums

if __package__:
    from . import refe_best_fit_seamless_support as support
else:
    import refe_best_fit_seamless_support as support


META_PIN = "META__FILE__fraglib.yaml"
SUPPORT_FILENAME = "refe_best_fit_seamless_support.py"
RESULT_CELLTYPE = "folder"


def seamlessify_refe_best_fit(
    rna_pdb: str | Path,
    fitter_arguments: Sequence[str] = (),
    *,
    fraglib_config: str | Path | None = None,
    conda_environment: str | None = None,
) -> Transformer:
    """Build and return one bash Transformer for the best-fit fitter.

    The raw YAML is supplied as a Seamless meta-file. Source, template, and
    input-PDB files are checksum-pinned through Seamless's native file loader;
    the library identity string comes only from its committed checksum
    sidecars. Caller code must initialize ``seamless_config`` before calling
    this function. The factory does not call ``seamless_config.init()``, execute
    the transformation, or materialize its result.
    """
    pdb = Path(rna_pdb).expanduser().resolve()
    config = (
        Path(fraglib_config).expanduser().resolve()
        if fraglib_config is not None
        else (Path.home() / ".alaric" / "fraglib.yaml").resolve()
    )
    arguments = tuple(fitter_arguments)
    if any(arg == "--pdb-output" or arg.startswith("--pdb-output=") for arg in arguments):
        raise ValueError(
            "--pdb-output is managed internally; pass fitter options only"
        )

    support_path = Path(support.__file__).resolve()
    input_paths = support.input_paths()
    input_paths.extend([support_path, pdb])

    input_names: dict[str, Path] = {}
    for path in input_paths:
        prior = input_names.get(path.name)
        if prior is not None:
            raise ValueError(
                f"Seamless flat input name collision: {prior} and {path} both map to {path.name!r}"
            )
        input_names[path.name] = path

    identities = json.dumps(
        support.identities(support_path.parent.parent / "fraglib"),
        sort_keys=True,
        separators=(",", ":"),
    )

    all_checksums, _ = files_to_checksums(
        [str(path) for path in input_names.values()] + [str(config)],
        directories={},
        max_upload_files=None,
        max_upload_size=None,
        auto_confirm="yes",
    )

    transformer = Transformer("bash")
    worker_command = shlex.join(
        ["python", "-B", SUPPORT_FILENAME, "run", pdb.name, *arguments]
    )
    transformer.code = (
        f"export fraglib_config=fraglib.yaml && {worker_command} "
        "&& mv best-fit-output RESULT"
    )
    for name, path in input_names.items():
        transformer.celltypes[name] = "bytes"
        transformer.args[name].checksum = Checksum(all_checksums[str(path)])

    transformer.celltypes[META_PIN] = "bytes"
    transformer.args[META_PIN].checksum = Checksum(all_checksums[str(config)])
    transformer.celltypes.fraglib_identities = "str"
    transformer.args.fraglib_identities = identities
    transformer.celltypes.result = RESULT_CELLTYPE
    if conda_environment is not None:
        transformer.environment.set_conda_env(conda_environment)

    return transformer
