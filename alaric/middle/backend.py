from __future__ import annotations

import shlex
import re
from pathlib import Path
from typing import Any

from .resolve import ResolvedAction, final_fragment, final_sequence
from .schema import OUTPUT_KIND


def q(value: str | Path | int | float) -> str:
    return shlex.quote(str(value))


def shell_path(value: str | Path) -> str:
    text = str(value)
    if "$" in text:
        return text
    return q(text)


_SHELL_REQUIRED_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*):\?\}")


def python_path(value: str | Path) -> str:
    text = _SHELL_REQUIRED_VAR.sub(r"${\1}", str(value))
    return repr(text)


def dep_result_path(dep: ResolvedAction, location: str) -> str:
    sigil = (dep.path / "sigil.txt").read_text().strip()
    if location == "remote":
        return f"${{ALARIC_REMOTE_RESULT_DIR}}/{sigil}"
    return f"../CACHE/results/{sigil}"


def result_file(dep: ResolvedAction, location: str) -> str:
    base = dep_result_path(dep, location)
    kind = OUTPUT_KIND[dep.action]
    if kind == "score":
        return f"{base}/score.npy"
    if kind == "mask":
        return f"{base}/mask.npy"
    return base


def data_file_path(filename: str, location: str) -> str:
    """Path to a DATA file param in a generated script.

    Local: read straight from the project's ``DATA/`` dir. Remote: the deployer uploads the
    file to a content-addressed global store (``$DEPLOYMENT_DIR/DATA/<checksum>``) and
    hardlinks it into the deployment dir under its filename, so the script references it
    relative to its own cwd.
    """
    if location == "remote":
        return q(f"./{filename}")
    return q(f"../DATA/{filename}")


def exclude_args(values: list[str], flag: str = "--pdb-exclude") -> str:
    if not values:
        return ""
    return " ".join([q(flag), *[q(v) for v in values]])


def score_exclude_args(values: list[str]) -> str:
    return " ".join(f"-x {q(v)}" for v in values)


def python_bin() -> str:
    return '"${PYTHON:-python}"'


def organize_command(alaric_dir: str, output_dir: str, location: str) -> str:
    # --compress and --max-poses-per-file are kept active (compression always on; max-poses
    # is the pinned, non-load-bearing layout knob). The remaining non-load-bearing knobs are
    # present but commented; uncommenting them never changes the canonical result.
    #
    # On remote (the pool lives on a network FS), --local-tempdir and --local-stagedir are
    # **active by default**: organize reads the unorganized shards from node-local scratch
    # (--local-tempdir) and writes the organized output to node-local scratch (--local-stagedir),
    # then bulk-moves it onto the FS — so NFS is touched only for the initial shard read and
    # the final move, not random I/O during bucketing. TMPDIR points that staging at
    # node-local scratch (size it for the decompressed input ~6 B/pose plus the organized
    # output). NB: --local-stagedir is non-atomic (a crash between deleting the unorganized
    # shards and committing the staged output loses both — fine for a regenerable chunk pool).
    remote = location == "remote"
    lines: list[str] = []
    if remote:
        lines.append('export TMPDIR="${ALARIC_REMOTE_SCRATCH_DIR:-${TMPDIR:-/tmp}}"')
    lines.append("# Non-load-bearing organize knobs (uncomment to tune; never change the result):")
    lines.append("organize_opts=(")
    if remote:
        lines.append("  --local-tempdir         # read shards from node-local scratch (avoids random NFS reads)")
        lines.append("  --local-stagedir        # write organized output to node-local scratch, then bulk-move to the FS")
    else:
        lines.append("#  --local-tempdir")
        lines.append("#  --local-stagedir")
    lines.append("#  --nprocs 8")
    lines.append("#  --capacity 500000000")
    lines.append("#  --chunk-poses 1000000")
    lines.append(")")
    lines.append(
        f"{python_bin()} {alaric_dir}/organize.py {shell_path(output_dir)} "
        '--compress --max-poses-per-file 100000000 ${organize_opts[@]+"${organize_opts[@]}"}'
    )
    return "\n".join(lines)


def score_concat_command(alaric_dir: str, output_dir: str, chunks_dir: str, location: str) -> str:
    # Memory-safe concat of the per-chunk score.npy files (memmap streaming, not
    # np.concatenate). On remote, point the staging dir at node-local scratch via TMPDIR so
    # the output file is built locally and then copied to the final (network FS) destination.
    lines: list[str] = []
    if location == "remote":
        lines.append('export TMPDIR="${ALARIC_REMOTE_SCRATCH_DIR:-${TMPDIR:-/tmp}}"')
    lines.append(
        f"{python_bin()} {alaric_dir}/score_concat.py {shell_path(chunks_dir)} "
        f"{shell_path(f'{output_dir}/score.npy')} --nchunks " + "{{ nchunks }}"
    )
    return "\n".join(lines)


def template_context(action: ResolvedAction, *, alaric_dir: str, output_dir: str, location: str) -> dict[str, Any]:
    p = action.params
    context: dict[str, Any] = {
        "action": action.action,
        "alaric_dir": alaric_dir,
        "python": python_bin(),
        "output_path": shell_path(output_dir),
        "output_path_python": python_path(output_dir),
    }
    if action.action in {"anchor", "anchor-test"}:
        dihedral = p["dihedral"]
        context.update(
            {
                "protein_path": data_file_path(p["__protein"], location),
                "resid": q(p["resid"]),
                "sequence": q(p["sequence"]),
                "dihedral_args": dihedral if isinstance(dihedral, str) else " ".join(q(v) for v in dihedral),
                "angle": q(p["angle"]),
                "margin": q(p.get("margin", 0.5)),
                "nucleotide_flag": "--first" if p["nucleotide"] == "first" else "--second",
                "nconformers": q(p.get("nconformers", "")),
                "exclude_args": exclude_args(p.get("exclude", [])),
                "exclude_python": repr(p.get("exclude", [])),
                "organize_command": organize_command(alaric_dir, output_dir, location),
            }
        )
    elif action.action == "grow":
        source = p["input"]
        context.update(
            {
                "input_result_path": shell_path(dep_result_path(source, location)),
                "input_result_python": python_path(dep_result_path(source, location)),
                "source_sequence": q(final_sequence(source)),
                "target_sequence": q(p["sequence"]),
                "direction": q(p["direction"]),
                "crmsd": q(p["crmsd"]),
                "ovrmsd": q(p["ovrmsd"]),
                "exclude_args": exclude_args(p.get("exclude", [])),
                "exclude_python": repr(p.get("exclude", [])),
                "organize_command": organize_command(alaric_dir, output_dir, location),
            }
        )
    elif action.action == "score":
        chunks_dir = f"{output_dir.removesuffix('.partial')}-CHUNKS"
        context.update(
            {
                "score_exclude_args": score_exclude_args(p.get("exclude", [])),
                "input_result_path": shell_path(dep_result_path(p["input"], location)),
                "input_result_python": python_path(dep_result_path(p["input"], location)),
                "sequence": q(p["sequence"]),
                "protein_path": data_file_path(p["__protein"], location),
                "nb_kernel": q(p.get("nb_kernel", "compiled")),
                "score_output_path": shell_path(f"{output_dir}/score.npy"),
                "score_chunks_path": shell_path(chunks_dir),
                "score_concat_command": score_concat_command(alaric_dir, output_dir, chunks_dir, location),
            }
        )
    elif action.action == "rmsd":
        context.update(
            {
                "input_result_path": shell_path(dep_result_path(p["input"], location)),
                "reference_path": data_file_path(p["__reference"], location),
                "fragment": q(p["fragment"]),
                "score_output_path": shell_path(f"{output_dir}/score.npy"),
                "exclude_args": exclude_args(p.get("exclude", [])),
            }
        )
    elif action.action == "score_add":
        context.update(
            {
                "score_input1_path": shell_path(result_file(p["score_input1"], location)),
                "score_input2_path": shell_path(result_file(p["score_input2"], location)),
                "score_output_path": shell_path(f"{output_dir}/score.npy"),
            }
        )
    elif action.action == "mask":
        context.update(
            {
                "score_input_path": shell_path(result_file(p["score_input"], location)),
                "threshold": q(p["threshold"]),
                "mask_output_path": shell_path(f"{output_dir}/mask.npy"),
            }
        )
    elif action.action == "filter":
        context["input_result_path"] = shell_path(dep_result_path(p["input"], location))
        context["filter_mode"] = "mask" if "mask_input" in p else "score"
        if "mask_input" in p:
            context["mask_input_path"] = shell_path(result_file(p["mask_input"], location))
            context["score_input_path"] = '""'
            context["threshold"] = '""'
        else:
            context["score_input_path"] = shell_path(result_file(p["score_input"], location))
            context["threshold"] = q(p["threshold"])
            context["mask_input_path"] = '""'
    elif action.action == "identity":
        context.update(
            {
                "input1_result_path": shell_path(dep_result_path(p["input1"], location)),
                "input2_result_path": shell_path(dep_result_path(p["input2"], location)),
            }
        )
    elif action.action == "bridge":
        input1 = p["input1"]
        input2 = p["input2"]
        lower = input1 if final_fragment(input1) == p["lower_fragment"] else input2
        upper = input1 if lower is input2 else input2
        context.update(
            {
                "lower_result_path": shell_path(dep_result_path(lower, location)),
                "upper_result_path": shell_path(dep_result_path(upper, location)),
                "lower_sequence": q(p["lower_sequence"]),
                "middle_sequence": q(p["middle_sequence"]),
                "upper_sequence": q(p["upper_sequence"]),
                "lower_crmsd": q(p["lower_crmsd"]),
                "lower_ovrmsd": q(p["lower_ovrmsd"]),
                "upper_crmsd": q(p["upper_crmsd"]),
                "upper_ovrmsd": q(p["upper_ovrmsd"]),
                "memory": q(p.get("memory", "600G")),
                "max_intermediate_poses": q(p.get("max-intermediate-poses", 100_000_000)),
                "max_final_poses": q(p.get("max-final-poses", 1_000_000)),
                "nprocs": q(p.get("nprocs", 1)),
                "rotamer_chunks": q(p.get("rotamer-chunks", 1)),
                "estimator_seed": q(p.get("estimator-seed", 0)),
                "estimator_sample_size": q(p.get("estimator-sample-size", 1000)),
            }
        )
    return context


def render_template(text: str, context: dict[str, Any]) -> str:
    rendered = text
    for key, value in context.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered
