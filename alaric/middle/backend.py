from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any

from .resolve import ResolvedAction, final_sequence
from .schema import OUTPUT_KIND


def q(value: str | Path | int | float) -> str:
    return shlex.quote(str(value))


def shell_path(value: str | Path) -> str:
    text = str(value)
    if "$" in text:
        return text
    return q(text)


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


def exclude_args(values: list[str], flag: str = "--pdb-exclude") -> str:
    if not values:
        return ""
    return " ".join([q(flag), *[q(v) for v in values]])


def score_exclude_args(values: list[str]) -> str:
    return " ".join(f"-x {q(v)}" for v in values)


def python_bin() -> str:
    return '"${PYTHON:-python}"'


def organize_command(alaric_dir: str, output_dir: str) -> str:
    return (
        f"{python_bin()} {alaric_dir}/organize.py {shell_path(output_dir)} "
        "--compress --max-poses-per-file 100000000 ${ALARIC_ORGANIZE_EXTRA_ARGS:-}"
    )


def template_context(action: ResolvedAction, *, alaric_dir: str, output_dir: str, location: str) -> dict[str, Any]:
    p = action.params
    context: dict[str, Any] = {
        "action": action.action,
        "alaric_dir": alaric_dir,
        "python": python_bin(),
        "output_path": shell_path(output_dir),
        "output_path_python": repr(output_dir),
    }
    if action.action in {"anchor", "anchor-test"}:
        dihedral = p["dihedral"]
        context.update(
            {
                "protein_path": q(f"../DATA/{p['__protein']}"),
                "resid": q(p["resid"]),
                "sequence": q(p["sequence"]),
                "dihedral_args": dihedral if isinstance(dihedral, str) else " ".join(q(v) for v in dihedral),
                "angle": q(p["angle"]),
                "margin": q(p.get("margin", 0.5)),
                "nucleotide_flag": "--first" if p["nucleotide"] == "first" else "--second",
                "nconformers": q(p.get("nconformers", "")),
                "exclude_args": exclude_args(p.get("exclude", [])),
                "exclude_python": repr(p.get("exclude", [])),
                "organize_command": organize_command(alaric_dir, output_dir),
            }
        )
    elif action.action == "grow":
        source = p["input"]
        context.update(
            {
                "input_result_path": shell_path(dep_result_path(source, location)),
                "input_result_python": repr(dep_result_path(source, location)),
                "source_sequence": q(final_sequence(source)),
                "target_sequence": q(p["sequence"]),
                "direction": q(p["direction"]),
                "crmsd": q(p["crmsd"]),
                "ovrmsd": q(p["ovrmsd"]),
                "exclude_args": exclude_args(p.get("exclude", [])),
                "exclude_python": repr(p.get("exclude", [])),
                "organize_command": organize_command(alaric_dir, output_dir),
            }
        )
    elif action.action == "score":
        context.update(
            {
                "score_exclude_args": score_exclude_args(p.get("exclude", [])),
                "input_result_path": shell_path(dep_result_path(p["input"], location)),
                "input_result_python": repr(dep_result_path(p["input"], location)),
                "sequence": q(p["sequence"]),
                "protein_path": q(f"../DATA/{p['__protein']}"),
                "nb_kernel": q(p.get("nb_kernel", "jax")),
                "score_output_path": shell_path(f"{output_dir}/score.npy"),
            }
        )
    elif action.action == "rmsd":
        context.update(
            {
                "input_result_path": shell_path(dep_result_path(p["input"], location)),
                "reference_path": q(f"../DATA/{p['__reference']}"),
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
    return context


def render_template(text: str, context: dict[str, Any]) -> str:
    rendered = text
    for key, value in context.items():
        rendered = rendered.replace("{{ " + key + " }}", str(value))
        rendered = rendered.replace("{{" + key + "}}", str(value))
    return rendered
