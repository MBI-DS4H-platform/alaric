"""Build the cached Seamless transformer for reference best-fit fitting.

Caller code must initialize ``seamless_config`` before using this factory.
The factory never calls ``seamless_config.init()`` or resets configuration;
execution and result materialization also remain with the caller.
"""
from __future__ import annotations

from pathlib import Path
import shlex
from collections.abc import Sequence
from math import isfinite, sqrt

from seamless import Checksum
from seamless_transformer import Transformer
from seamless_transformer.cmd.file_load import files_to_checksums

META_PIN = "META__FILE__fraglib.yaml"
SUPPORT_FILENAME = "refe_best_fit_seamless_support.py"
RESULT_CELLTYPE = "folder"
DEPS_FILENAME = "refe-best-fit.DEPS"
GRID_SPACING = sqrt(3) / 3
ROTAMER_PRECISION = 0.5
_LENIENCY_OR_FORMAT_FLAGS = {
    "--ignore-unknown",
    "--ignore-missing",
    "--ignore-reordered",
    "--no-header",
}
_SUPPORTED_FITTER_OPTIONS = {"--grid-spacing", "--rotamer-precision"}


def _validate_fitter_arguments(arguments: tuple[str, ...]) -> None:
    if any(not isinstance(argument, str) for argument in arguments):
        raise TypeError("fitter arguments must be strings")

    index = 0
    while index < len(arguments):
        argument = arguments[index]
        option = argument.split("=", 1)[0]
        if option == "--pdb-output":
            raise ValueError(
                "--pdb-output is managed internally; pass fitter options only"
            )
        if option in _LENIENCY_OR_FORMAT_FLAGS:
            raise ValueError(f"{option} is forbidden by the T5 input/output contract")

        # T5 permits only the two fixed numeric fitter options below. Reject
        # abbreviations and unknown options here, before argparse could resolve
        # an abbreviated spelling such as --no-head or --exclude-p.
        if option not in _SUPPORTED_FITTER_OPTIONS:
            raise ValueError(f"Unsupported fitter option: {option}")

        if "=" in argument:
            value = argument.split("=", 1)[1]
            consumed = 1
        else:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            value = arguments[index + 1]
            consumed = 2
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{option} must be numeric") from exc
        required = GRID_SPACING if option == "--grid-spacing" else ROTAMER_PRECISION
        if not isfinite(number) or number != required:
            if option == "--grid-spacing":
                raise ValueError(
                    f"--grid-spacing is fixed at sqrt(3)/3 ({GRID_SPACING!r})"
                )
            raise ValueError(
                f"--rotamer-precision is fixed at {ROTAMER_PRECISION:g}"
            )
        index += consumed


def seamlessify_refe_best_fit(
    fitter_arguments: Sequence[str] = (),
    *,
    fraglib_config: str | Path | None = None,
    conda_environment: str | None = None,
) -> Transformer:
    """Build and return a bash Transformer with unbound project inputs.

    The raw YAML is supplied as a Seamless meta-file. Source, template, and
    fixed data files are checksum-pinned through Seamless's native file loader.
    ``rna_pdb``, ``exclude_pdb_code`` and ``fraglib_identities`` remain
    unbound project input pins for the caller to connect after construction.
    Caller code must initialize ``seamless_config`` before calling this
    function. The factory does not call ``seamless_config.init()``, execute the
    transformation, or materialize its result.
    """
    config = (
        Path(fraglib_config).expanduser().resolve()
        if fraglib_config is not None
        else (Path.home() / ".alaric" / "fraglib.yaml").resolve()
    )
    arguments = tuple(fitter_arguments)
    _validate_fitter_arguments(arguments)
    if any(
        arg in ("--exclude", "--exclude-pdb-code")
        or arg.startswith("--exclude=")
        or arg.startswith("--exclude-pdb-code=")
        for arg in arguments
    ):
        raise ValueError(
            "--exclude-pdb-code is a connectable pin; set exclude_pdb_code after construction"
        )

    util_dir = Path(__file__).resolve().parent
    deps_path = util_dir / DEPS_FILENAME
    dependencies = deps_path.read_text(encoding="utf-8").splitlines()
    if not dependencies or any(
        not item or item.startswith("#") for item in dependencies
    ):
        raise ValueError(
            f"{DEPS_FILENAME} must contain one non-empty path per line and no comments"
        )
    input_paths = [deps_path]
    input_paths.extend((util_dir / item).resolve() for item in dependencies)

    input_names: dict[str, Path] = {}
    for path in input_paths:
        prior = input_names.get(path.name)
        if prior is not None:
            raise ValueError(
                f"Seamless flat input name collision: {prior} and {path} both map to {path.name!r}"
            )
        input_names[path.name] = path

    all_checksums, _ = files_to_checksums(
        [str(path) for path in input_names.values()] + [str(config)],
        directories={},
        max_upload_files=None,
        max_upload_size=None,
        auto_confirm="yes",
    )

    transformer = Transformer("bash")
    worker_command = shlex.join(
        ["python", "-B", SUPPORT_FILENAME, "run", *arguments]
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
    transformer.celltypes.rna_pdb = "bytes"
    transformer.celltypes.exclude_pdb_code = "str"
    transformer.celltypes.fraglib_identities = "str"
    transformer.celltypes.result = RESULT_CELLTYPE
    if conda_environment is not None:
        transformer.environment.set_conda_env(conda_environment)

    return transformer
