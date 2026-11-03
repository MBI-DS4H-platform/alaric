"""A9 end-to-end oracles over the accepted A7/A8 installation."""
from __future__ import annotations

import csv
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "tests" / "1b7f"
TESTS = ROOT / "tests"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from alaric import gaussian_cache, gaussian_filter  # noqa: E402
from gaussian_common import load_spec, require_caches  # noqa: E402
from w6_gaussian_spawn import run_check as _run_check_in_spawn  # noqa: E402


def _fraglib_identities() -> str:
    from util.refe_best_fit_seamless_support import identities

    return json.dumps(
        identities(ROOT / "fraglib"), sort_keys=True, separators=(",", ":")
    )


def _pdb_atom(serial: int, atom: str, residue: str, resid: int, xyz: tuple[float, float, float]) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {atom:>4s} {residue:>3s} A{resid:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          C \n"
    )


def _prepare_case(root: Path) -> tuple[Path, Path, list[dict], str]:
    best_fit = root / "best-fit"
    (best_fit / "pdbs").mkdir(parents=True)
    shutil.copyfile(HERE / "data" / "refe-best-fit.tsv", best_fit / "best-fits.tsv")
    for path in (HERE / "best-fits").glob("*-bestfit.pdb"):
        shutil.copyfile(path, best_fit / "pdbs" / path.name)

    proteins = root / "proteins"
    proteins.mkdir()
    # These 1b7f structures retain the source residue numbering expected by the
    # T20 job fields. Aareduce-compatible ring atoms are parsed by A9.
    shutil.copyfile(HERE / "pdbs" / "1b7f_dom2.pdb", proteins / "range-2-aa.pdb")
    shutil.copyfile(HERE / "pdbs" / "1b7f_dom1.pdb", proteins / "range-3-aa.pdb")
    # One intentionally incomplete PHE ring exercises row-local failure handling.
    (proteins / "range-4-aa.pdb").write_text(
        "HEADER incomplete ring fixture\n"
        + _pdb_atom(1, "CG", "PHE", 777, (0.0, 0.0, 0.0))
        + "END\n"
    )

    with (best_fit / "best-fits.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    contacts = [(2, 214), (2, 256), (3, 131), (3, 170)]
    jobs = []
    # Twenty distinct T20-shaped rows cover both nucleotide positions and the
    # four checked protein contacts in the delivered 1b7f fixture.
    for index in range(20):
        row = rows[(index * 7) % len(rows)]
        protein_range, protein_resid = contacts[index % len(contacts)]
        jobs.append(
            {
                "key": f"w6-1b7f-{index:02d}",
                "fragment": int(row["fragment"]),
                "nucleotide": "first" if index % 2 == 0 else "second",
                "sequence": row["sequence"],
                "conformer": int(row["conformer"]),
                "rotamer": int(row["rotamer"]),
                "grid": [int(row[f"grid_{axis}"]) for axis in "xyz"],
                "range": protein_range,
                "protein_resid": protein_resid,
            }
        )

    jobs.extend(
        [
            {
                **jobs[0],
                "key": "w6-incomplete-protein-ring",
                "range": 4,
                "protein_resid": 777,
            },
            {
                **jobs[1],
                "key": "w6-reconstruction-error",
                "grid": [jobs[1]["grid"][0] + 20, *jobs[1]["grid"][1:]],
            },
        ]
    )
    return best_fit, proteins, jobs, _fraglib_identities()


@pytest.fixture(scope="module")
def spawned_result(tmp_path_factory):
    require_caches(gaussian_filter.all_names())
    work = tmp_path_factory.mktemp("w6-a9")
    best_fit, proteins, jobs, identities = _prepare_case(work)
    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    payload = (jobs, str(best_fit), str(proteins), "1b7f", identities)
    # The child drops any Seamless service env before importing A9. Its module,
    # array and library state is cold by construction under multiprocessing spawn.
    process = ctx.Process(
        target=_run_check_in_spawn,
        args=(send, (*payload, ROOT)),
    )
    process.start()
    send.close()
    process.join(timeout=600)
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("cold spawned A9 run exceeded the 600 s test safety limit")
    assert process.exitcode == 0
    assert receive.poll(5), "spawned worker exited without an A9 result"
    status, value = receive.recv()
    receive.close()
    assert status == "ok", value
    return value


def test_a9_spawned_result_has_the_twelve_json_defined_filters(spawned_result) -> None:
    spec = load_spec()
    names = set(spec["thresholds"])
    assert names == set(gaussian_filter.all_names())
    assert spawned_result["schema"] == 1
    assert spawned_result["filter_set_version"] == spec["filter_set_version"]
    assert spawned_result["min_cell_fraction"] == 1 / 3
    assert len(spawned_result["jobs"]) == 22

    by_key = {job["key"]: job for job in spawned_result["jobs"]}
    assert set(by_key) == {f"w6-1b7f-{i:02d}" for i in range(20)} | {
        "w6-incomplete-protein-ring",
        "w6-reconstruction-error",
    }
    for job in spawned_result["jobs"]:
        if "error" in job:
            continue
        assert set(job["filters"]) == names
        for row in job["filters"].values():
            if "implied_by" in row:
                assert row == {"emitted": False, "implied_by": row["implied_by"]}
                assert row["implied_by"] in names
            else:
                assert {"count", "capacity", "fill", "emitted", "pose_pass", "score"} <= set(row)
                assert row["capacity"] > 0
                assert row["fill"] == pytest.approx(row["count"] / row["capacity"])
                assert row["emitted"] is (row["fill"] >= 1 / 3)


def test_a9_containment_and_implied_rejections_are_explicit(spawned_result) -> None:
    spec = load_spec()
    names = set(spec["thresholds"])
    contained_by = gaussian_filter.containment_edges(spec["thresholds"])
    seen_implied = False
    for job in spawned_result["jobs"]:
        if "error" in job:
            continue
        filters = job["filters"]
        for strict, broad in contained_by:
            strict_result = filters[strict]
            broad_result = filters[broad]
            if "implied_by" in strict_result:
                seen_implied = True
                assert strict_result["emitted"] is False
                assert (strict, strict_result["implied_by"]) in contained_by
                assert set(strict_result) == {"emitted", "implied_by"}
                continue
            if "implied_by" not in broad_result:
                assert strict_result["emitted"] <= broad_result["emitted"]
                assert strict_result["pose_pass"] <= broad_result["pose_pass"]
                assert strict_result["count"] <= broad_result["count"]
                assert strict_result["capacity"] == broad_result["capacity"]
    assert names == set(gaussian_filter.all_names())
    assert seen_implied, "fixture must exercise broad-to-strict implied rejection rows"


def test_a9_reports_bad_rings_and_bad_reconstructions_per_job(spawned_result) -> None:
    jobs = {job["key"]: job for job in spawned_result["jobs"]}
    assert "error" in jobs["w6-incomplete-protein-ring"]
    assert "error" in jobs["w6-reconstruction-error"]
    reconstruction_error = str(jobs["w6-reconstruction-error"]["error"]).lower()
    assert "reconstruction" in reconstruction_error or "rmsd" in reconstruction_error
    assert "error" not in jobs["w6-1b7f-00"]


def test_semantic_cache_identity_ignores_compression_but_tracks_arrays_and_spec() -> None:
    from alaric.gaussian_check import semantic_cache_identity

    spec = load_spec()
    original = semantic_cache_identity(spec)
    recompressed = copy.deepcopy(spec)
    for entry in recompressed["thresholds"].values():
        for file_entry in entry["precalc"]["files"].values():
            file_entry["sha256"] = "0" * 64
    assert semantic_cache_identity(recompressed) == original

    changed_array = copy.deepcopy(spec)
    first = next(iter(changed_array["thresholds"]))
    first_file = next(iter(changed_array["thresholds"][first]["precalc"]["files"]))
    changed_array["thresholds"][first]["precalc"]["files"][first_file]["npy_sha256"] = "f" * 64
    assert semantic_cache_identity(changed_array) != original

    changed_spec = copy.deepcopy(spec)
    changed_spec["spec_digest"] = "f" * 64
    assert semantic_cache_identity(changed_spec) != original


def test_a9_runtime_checks_the_separate_semantic_identity_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from alaric import gaussian_check
    from alaric.gaussian_check import semantic_cache_identity

    best_fit = tmp_path / "best-fit"
    proteins = tmp_path / "proteins"
    best_fit.mkdir()
    proteins.mkdir()
    spec = load_spec()
    identity = semantic_cache_identity(spec)

    monkeypatch.setattr(gaussian_check, "load_spec", lambda _path: spec)
    result = gaussian_check.run_gaussian_check(
        [], best_fit, proteins, "1b7f", _fraglib_identities(), ROOT,
        semantic_identity=identity,
    )
    assert result["jobs"] == []

    recompressed = copy.deepcopy(spec)
    for entry in recompressed["thresholds"].values():
        for file_entry in entry["precalc"]["files"].values():
            file_entry["sha256"] = "0" * 64
    monkeypatch.setattr(gaussian_check, "load_spec", lambda _path: recompressed)
    assert gaussian_check.run_gaussian_check(
        [], best_fit, proteins, "1b7f", _fraglib_identities(), ROOT,
        semantic_identity=identity,
    )["jobs"] == []

    changed_array = copy.deepcopy(recompressed)
    first_name = next(iter(changed_array["thresholds"]))
    first_file = next(iter(changed_array["thresholds"][first_name]["precalc"]["files"]))
    changed_array["thresholds"][first_name]["precalc"]["files"][first_file][
        "npy_sha256"
    ] = "f" * 64
    monkeypatch.setattr(gaussian_check, "load_spec", lambda _path: changed_array)
    with pytest.raises(ValueError, match="semantic_identity"):
        gaussian_check.run_gaussian_check(
            [], best_fit, proteins, "1b7f", _fraglib_identities(), ROOT,
            semantic_identity=identity,
        )


def test_a9_semantic_identity_participates_in_seamless_transform_checksum(
    tmp_path: Path,
) -> None:
    import importlib.util

    if importlib.util.find_spec("seamless_config") is None:
        pytest.skip("seamless_config is unavailable in this environment")

    from seamless_config.select import (
        get_execution,
        get_persistent,
        select_execution,
        select_persistent,
    )
    from seamless_transformer import delayed

    from alaric.gaussian_check import run_gaussian_check, semantic_cache_identity

    previous_execution, previous_persistent = get_execution(), get_persistent()
    select_execution("process")
    select_persistent(False)
    best_fit = tmp_path / "best-fit"
    proteins = tmp_path / "proteins"
    best_fit.mkdir()
    proteins.mkdir()
    spec = load_spec()
    identity = semantic_cache_identity(spec)
    fraglib_identities = {
        "schema": 1,
        "fraglen": 2,
        "sha256": {"fixture": "0" * 64},
    }

    def transform_checksum(semantic_identity: dict, jobs: list[dict]):
        transformation = delayed(run_gaussian_check).build(
            jobs=jobs,
            best_fit=str(best_fit),
            proteins=str(proteins),
            exclude_pdb_code="1b7f",
            fraglib_identities=fraglib_identities,
            repo_root=str(ROOT),
            semantic_identity=semantic_identity,
        )
        transformation.construct()
        return transformation.transformation_checksum

    try:
        baseline_jobs = [{"key": "baseline"}]
        original_checksum = transform_checksum(identity, baseline_jobs)
        recompressed = copy.deepcopy(spec)
        for entry in recompressed["thresholds"].values():
            for file_entry in entry["precalc"]["files"].values():
                file_entry["sha256"] = "0" * 64
        assert (
            transform_checksum(semantic_cache_identity(recompressed), baseline_jobs)
            == original_checksum
        )

        changed_array_identity = copy.deepcopy(identity)
        first_name = next(iter(changed_array_identity["arrays"]))
        first_file = next(iter(changed_array_identity["arrays"][first_name]))
        changed_array_identity["arrays"][first_name][first_file] = "f" * 64
        assert transform_checksum(changed_array_identity, baseline_jobs) != original_checksum
        changed_spec_identity = copy.deepcopy(identity)
        changed_spec_identity["spec_digest"] = "f" * 64
        assert transform_checksum(changed_spec_identity, baseline_jobs) != original_checksum
        assert transform_checksum(
            identity, [{"key": "different-runtime-input"}]
        ) != original_checksum
    finally:
        select_execution(previous_execution)
        select_persistent(previous_persistent)


def _recompress_verified_file(source: Path, destination: Path) -> None:
    """Write the exact logical .npy bytes with a fresh zstd representation."""
    import zstandard

    destination.parent.mkdir(parents=True, exist_ok=True)
    for level in (1, 3, 9, 19, 22):
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with source.open("rb") as source_handle:
            if source.name.endswith(".zst"):
                reader = zstandard.ZstdDecompressor().stream_reader(source_handle)
            else:
                reader = source_handle
            with reader, temporary.open("wb") as target_handle:
                zstandard.ZstdCompressor(level=level).copy_stream(
                    reader, target_handle
                )
        if (
            source.name.endswith(".zst")
            and gaussian_cache._sha_file(temporary)
            == gaussian_cache._sha_file(source)
        ):
            continue
        temporary.replace(destination)
        return
    temporary.unlink(missing_ok=True)
    raise AssertionError(f"could not produce a new compressed representation for {source}")


def test_a9_recompressed_cache_preserves_semantic_identity_and_decision(
    tmp_path: Path,
    spawned_result: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rerun the broadest branch against newly compressed, byte-identical arrays."""
    from alaric import gaussian_check
    from alaric.gaussian_cache import verify_file
    from alaric.gaussian_check import semantic_cache_identity

    by_key = {job["key"]: job for job in spawned_result["jobs"]}
    selected = next(
        job
        for job in by_key.values()
        if "error" not in job and not job["filters"]["rrm95"]["emitted"]
    )
    best_fit, proteins, jobs, identities = _prepare_case(tmp_path / "case")
    job = next(item for item in jobs if item["key"] == selected["key"])

    spec = load_spec()
    original_identity = semantic_cache_identity(spec)
    filter_entry = spec["thresholds"]["rrm95"]["precalc"]
    source_paths = gaussian_cache.verify_filter_cache(spec, "rrm95", ROOT)
    temp_root = tmp_path / "recompressed-repo"
    temp_alaric = temp_root / "alaric"
    temp_alaric.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "alaric" / "gaussian_stacking_filter.json",
        temp_alaric / "gaussian_stacking_filter.json",
    )
    shutil.copytree(ROOT / "ring-refe", temp_root / "ring-refe")
    temp_spec = gaussian_filter.load_spec(
        temp_alaric / "gaussian_stacking_filter.json"
    )
    cache_dir = temp_root / filter_entry["dir"]
    cache_dir.mkdir(parents=True)
    for filename, source_path in source_paths.items():
        _recompress_verified_file(
            Path(source_path), cache_dir / f"{filename}.npy.zst"
        )
        expected = filter_entry["files"][filename]
        resolved, status = verify_file(
            cache_dir / f"{filename}.npy",
            expected.get("sha256"),
            expected["npy_sha256"],
        )
        assert resolved.suffix == ".zst"
        assert status == "recompressed"

    assert semantic_cache_identity(temp_spec) == original_identity
    # The broadest filter rejects this row, so the remaining eleven filters are
    # explicitly implied and the isolated temporary repo needs only rrm95 data.
    monkeypatch.setattr(gaussian_check, "_MODULE_DIR", temp_alaric)
    rerun = gaussian_check.run_gaussian_check(
        [job], best_fit, proteins, "1b7f", identities, temp_root
    )
    assert rerun["jobs"][0] == selected
    assert rerun["jobs"][0]["filters"]["rrm95"]["emitted"] is False


@pytest.mark.w6_cold_timing
@pytest.mark.skipif(
    os.environ.get("ALARIC_RUN_W6_COLD_TIMING") != "1",
    reason="explicit acceptance run; prepare a temporary .npy cache install first",
)
def test_a9_twenty_jobs_cold_spawn_under_five_seconds(tmp_path_factory) -> None:
    """Timed acceptance; run only against a temporary, pre-decompressed cache root."""
    require_caches(gaussian_filter.all_names())
    work = tmp_path_factory.mktemp("w6-cold-timing")
    best_fit, proteins, jobs, identities = _prepare_case(work)
    cache_root = work / "repo"
    (cache_root / "alaric").mkdir(parents=True)
    shutil.copyfile(
        ROOT / "alaric" / "gaussian_stacking_filter.json",
        cache_root / "alaric" / "gaussian_stacking_filter.json",
    )
    shutil.copytree(ROOT / "ring-refe", cache_root / "ring-refe")
    cache_dir = ROOT / "precomputed-filters" / "gaussian"
    destination = cache_root / "precomputed-filters" / "gaussian"
    shutil.copytree(cache_dir, destination)
    for child in destination.iterdir():
        gaussian_cache.decompress_dir(child)

    ctx = mp.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    payload = (jobs[:20], str(best_fit), str(proteins), "1b7f", identities)
    process = ctx.Process(target=_run_check_in_spawn, args=(send, (*payload, cache_root)))
    import time

    started = time.perf_counter()
    process.start()
    send.close()
    process.join(timeout=60)
    elapsed = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        pytest.fail("cold spawned 20-job A9 run exceeded 60 s")
    assert process.exitcode == 0
    assert receive.poll(5)
    status, value = receive.recv()
    assert status == "ok", value
    assert len(value["jobs"]) == 20
    print(
        "A9 cold acceptance: jobs=20 cache=.npy spawn=fresh conda=alaric "
        f"elapsed={elapsed:.3f}s"
    )
    assert elapsed <= 5.0, f"cold spawned A9 took {elapsed:.3f}s for 20 jobs"
