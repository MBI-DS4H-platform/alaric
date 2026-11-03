from __future__ import annotations

import json
import shlex
import re
from pathlib import Path
from typing import Any

from .resolve import ResolvedAction, final_sequence
from .schema import OUTPUT_KIND


# Node-local scratch, with the same fallback chain the organize staging already uses.
SCRATCH_EXPR = "${ALARIC_REMOTE_SCRATCH_DIR:-${TMPDIR:-/tmp}}"
# Set by a wholesale run.sh to redirect an action's intermediate output onto node-local
# scratch. Unset (the default) keeps that output on the shared result FS, which is what
# independently submitted chunkN.sh jobs require: they run on different nodes than the
# organize job, so a node-local pool would be invisible to it.
POOL_ENV_VAR = "ALARIC_UNORGANIZED_DIR"


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


def python_bin() -> str:
    return '"${PYTHON:-python}"'


def organize_command(alaric_dir: str, output_dir: str, location: str, *, pool: str = "shared") -> str:
    """Render the organize invocation for a pose-producing action.

    --compress and --max-poses-per-file are kept active (compression always on; max-poses
    is the pinned, non-load-bearing layout knob). The remaining non-load-bearing knobs are
    present but commented; uncommenting them never changes the canonical result.

    ``pool`` says where the unorganized shards live, which decides the staging knobs:

    * ``shared`` — on the network FS, because sibling chunks and organize.sh run as
      independent jobs on different nodes. --local-tempdir and --local-stagedir are then
      **active by default**: organize reads the shards through node-local scratch and
      writes the organized output there too, bulk-moving it onto the FS at the end — so
      the FS sees one bulk read and one bulk write instead of random I/O during bucketing.
      NB: --local-stagedir is non-atomic (a crash between deleting the unorganized shards
      and committing the staged output loses both — fine for a regenerable chunk pool).
    * ``local`` — already on node-local scratch, because a single script produced the whole
      pool itself. Both knobs are then dead weight: --local-tempdir would copy and
      decompress local→local (~4× scratch, ~6 B/pose) and --local-stagedir would stage a
      local→local move. They stay commented.
    * ``auto`` — one script serves both cases (the chunk deployer's organize.sh), so the
      choice is deferred to runtime and keyed on POOL_ENV_VAR.

    TMPDIR points any staging that does happen at node-local scratch.
    """
    remote = location == "remote"
    lines: list[str] = []
    if remote:
        lines.append(f'export TMPDIR="{SCRATCH_EXPR}"')
    lines.append("# Non-load-bearing organize knobs (uncomment to tune; never change the result):")
    lines.append("organize_opts=(")
    if remote and pool == "shared":
        lines.append("  --local-tempdir         # read shards from node-local scratch (avoids random NFS reads)")
        lines.append("  --local-stagedir        # write organized output to node-local scratch, then bulk-move to the FS")
    else:
        lines.append("#  --local-tempdir")
        lines.append("#  --local-stagedir")
    lines.append("#  --nprocs 8")
    lines.append("#  --capacity 500000000")
    lines.append("#  --chunk-poses 1000000")
    lines.append(")")
    if remote and pool == "auto":
        lines.extend(
            [
                f'if [ -z "${{{POOL_ENV_VAR}:-}}" ]; then',
                "  # Shards are on the shared FS: stage the reads and the organized writes",
                "  # through node-local scratch so the FS sees bulk instead of random I/O.",
                "  organize_opts+=(--local-tempdir --local-stagedir)",
                "fi",
            ]
        )
    lines.append(
        f"{python_bin()} {alaric_dir}/organize.py {shell_path(output_dir)} "
        '--compress --max-poses-per-file 100000000 ${organize_opts[@]+"${organize_opts[@]}"}'
    )
    return "\n".join(lines)


def conformer_range_args(p: dict[str, Any]) -> str:
    if "conformer" in p:
        conformer = q(p["conformer"])
        return f"--conformer-range {conformer} {conformer}"
    return f"--conformer-range 1 {q(p['nconformers'])}"


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
        if not p.get("__precomputed_filter_dir"):
            raise ValueError("anchor requires a precomputed filter (precomputed_filter_name/threshold)")
        precalc_dir = p["__precomputed_filter_dir"]
        if location == "remote":
            # On remote, precomputed filter directory and reference rings are in alaric repo
            precalc_dir = f"${{ALARIC_REMOTE_ALARIC_DIR:?}}/../{p['__precomputed_filter_relative_dir']}"
            ref_ring = f"${{ALARIC_REMOTE_ALARIC_DIR:?}}/../ring-refe/{p['__anchor_reference_ring']}"
            protein_ref_ring = f"${{ALARIC_REMOTE_ALARIC_DIR:?}}/../ring-refe/{p['__anchor_reference_protein_ring']}"
        else:
            # On local, reference rings are in ALARIC_DIR/../ring-refe/
            ref_ring = f"${{ALARIC_DIR:?}}/../ring-refe/{p['__anchor_reference_ring']}"
            protein_ref_ring = f"${{ALARIC_DIR:?}}/../ring-refe/{p['__anchor_reference_protein_ring']}"
        precalc_args = (
            f"--precalculated-anchor {shell_path(precalc_dir)} "
            f"--precalculated-filter-name {q(p['precomputed_filter_threshold'])} "
            f"--anchor-reference-ring {ref_ring} "
            f"--anchor-reference-protein-ring {protein_ref_ring}"
        )
        context.update(
            {
                "protein_path": data_file_path(p["__protein"], location),
                "protein_range": q(p["protein"]),
                "resid": q(p["resid"]),
                "sequence": q(p["sequence"]),
                "nucleotide_flag": "--first" if p["nucleotide"] == "first" else "--second",
                "nconformers": q(p.get("nconformers", "")),
                "single_conformer": q(p["conformer"]) if "conformer" in p else "",
                "conformer_range_args": conformer_range_args(p) if action.action == "anchor-test" else "",
                "exclude_args": exclude_args(p.get("exclude", [])),
                "exclude_python": repr(p.get("exclude", [])),
                "precalculated_args": precalc_args,
                "organize_command": organize_command(alaric_dir, output_dir, location),
            }
        )
    elif action.action == "anchor-refe":
        context.update(
            {
                "reference_path": data_file_path(p["__reference"], location),
                "fragment": q(p["fragment"]),
                "sequence": q(p["sequence"]),
                "nucleotide_flag": "--first" if p["nucleotide"] == "first" else "--second",
                "ovrmsd": q(p["ovrmsd"]),
                "exclude_args": exclude_args(p.get("exclude", [])),
                "exclude_python": repr(p.get("exclude", [])),
                "organize_command": organize_command(alaric_dir, output_dir, location),
            }
        )
    elif action.action in {"grow", "grow-test"}:
        source = p["input"]
        restrict = p.get("restrict_input")
        restrict_args = ""
        if restrict is not None:
            restrict_args = (
                "--restrict-poses "
                + shell_path(dep_result_path(restrict, location))
            )
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
                "restrict_args": restrict_args,
                "conformer_args": f"--conformer {q(p['conformer'])}" if action.action == "grow-test" else "",
                "organize_command": organize_command(alaric_dir, output_dir, location),
            }
        )
    elif action.action == "energy-filter":
        cascade_json = json.dumps(p["cascade"], separators=(",", ":"))
        context.update(
            {
                "input_result_path": shell_path(dep_result_path(p["input"], location)),
                "protein_path": data_file_path(p["__protein"], location),
                "protein_range": q(p["protein"]),
                "sequence": q(p["sequence"]),
                "fragment": q(p["fragment"]),
                "exclude_args": exclude_args(p.get("exclude", [])),
                "threshold": q(p["threshold"]),
                "epsilon": q(p["epsilon"]),
                "cascade_json": q(cascade_json),
                "min_cascade": q(p["min_cascade"]),
                "nprocs": q(p.get("nprocs", 1)),
                "chunk_poses": q(p.get("chunk_poses", 10000)),
                "output_path": shell_path(output_dir),
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
    elif action.action == "mask":
        context.update(
            {
                "score_input_path": shell_path(result_file(p["score_input"], location)),
                "threshold": q(p["threshold"]),
                "mask_output_path": shell_path(f"{output_dir}/mask.npy"),
            }
        )
    elif action.action == "identity":
        context.update(
            {
                "input1_result_path": shell_path(dep_result_path(p["input1"], location)),
                "input2_result_path": shell_path(dep_result_path(p["input2"], location)),
            }
        )
    elif action.action == "mask-common-conformer":
        context.update(
            {
                "input1_result_path": shell_path(dep_result_path(p["input1"], location)),
                "input2_result_path": shell_path(dep_result_path(p["input2"], location)),
            }
        )
    return context


def render_template(text: str, context: dict[str, Any]) -> str:
    rendered = text
    for key, value in context.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered
