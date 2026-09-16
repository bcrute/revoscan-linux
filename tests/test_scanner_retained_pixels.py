"""M14 exact-count retained-pixel admission and scoped adapter provenance."""

import hashlib
import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scanner3d import icp_plane, tracking
from scanner3d.depth_metric import ensure_intrinsics_usable
from scanner3d.geometry import Rigid
from scanner3d.simulate import FACTORY_INTRINSICS
from scanner3d.tracking import DEFAULT_POLICY, IncrementalTracker


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "tools"))
    return importlib.import_module("scanner_retained_pixels")


def cloud(size):
    fx,fy,cx,cy = ensure_intrinsics_usable(FACTORY_INTRINSICS)
    return tuple(((i % 80 + 200 - cx) * (100.+i%17*.1)/fx,
                  (i // 80 + 40 - cy) * (100.+i%17*.1)/fy,100.+i%17*.1)
                 for i in range(size))


@pytest.mark.parametrize("size", [899,900,901,1699,1800])
@pytest.mark.parametrize("profile", ["retained-stride","retained-pixel-grid"])
def test_exact_count_order_pose_admission_and_original_object(module,monkeypatch,size,profile):
    points = cloud(size)
    value = module.RetainedPixelTracker(backend="scipy",policy=module.EXPECTED_POLICY,
                                       retention_profile=profile)
    control = IncrementalTracker(backend="scipy",policy=module.EXPECTED_POLICY)
    before = module.ORIGINAL_SAMPLED
    samples = []

    def original(p,limit,*,offset=0):
        result = before(p,limit,offset=offset)
        samples.append(result)
        return result

    poses = (Rigid.identity(),replace(Rigid.identity(),translation=(1.,2.,3.)))
    for index,pose in enumerate(poses):
        control._offers_since_accepted = value._offers_since_accepted = 1
        expected = control._admit(index,points,pose,overlap=1.,anchor=index==0)
        with monkeypatch.context() as scoped:
            scoped.setattr(module,"ORIGINAL_SAMPLED",original)
            scoped.setattr(tracking,"_sampled",original)
            actual = value._admit(index,points,pose,overlap=1.,anchor=index==0)
        assert actual == expected and value._velocity == control._velocity
        assert value.keyframes == control.keyframes
        assert value._accepted[index].pose is pose
        recorded = value.retention_records[-1]
        assert recorded["sampler_calls"] == 1 and recorded["status"] == "completed"
        assert len(value._accepted[index].cloud) == len(control._accepted[index].cloud)
        assert recorded["returned_points"] == recorded["original_retained_points"]
        if profile == "retained-stride":
            assert value._accepted[index].cloud is samples[-1]
            assert value._accepted[index].cloud == control._accepted[index].cloud
        else:
            selected = module.indices(points,len(control._accepted[index].cloud),capacity=1800)
            assert value._accepted[index].cloud == tuple(points[i] for i in selected)
            assert selected == tuple(sorted(selected))
        assert not value._inside_retention
    assert tracking._sampled is before
    for name in ("offer","_candidate","_fit","_model_reference","_reference_frames",
                 "_update_keyframes","_release_clouds"):
        assert getattr(type(value),name) is getattr(IncrementalTracker,name)


@pytest.mark.parametrize("failure", ["oversized","duplicate","nonpixel","nonfinite","empty"])
def test_bad_native_provenance_fails_before_admission_mutates(module,failure):
    points = cloud(1700)
    if failure == "oversized":
        points = cloud(1801)
    elif failure == "duplicate":
        points = (*points[:-1],points[0])
    elif failure == "nonpixel":
        p = points[0]
        points = ((p[0]+.1,p[1],p[2]),*points[1:])
    elif failure == "nonfinite":
        points = ((float('nan'),0.,100.),*points[1:])
    else:
        points = ()
    value = module.RetainedPixelTracker(backend="scipy",policy=module.EXPECTED_POLICY,
                                       retention_profile="retained-pixel-grid")
    value._offers_since_accepted = 7
    with pytest.raises((ValueError,RuntimeError)):
        value._admit(0,points,Rigid.identity(),overlap=1.,anchor=True)
    assert value._accepted == [] and value.keyframes == ()
    assert value._offers_since_accepted == 7 and value._velocity is None
    assert not value._inside_retention and tracking._sampled is module.ORIGINAL_SAMPLED
    assert value.retention_records[0]["status"] == "failed"


@pytest.mark.parametrize("retention", ["retained-stride","retained-pixel-grid"])
@pytest.mark.parametrize("normal", ["observe-10","normal-20"])
@pytest.mark.parametrize("static", [False,True])
@pytest.mark.parametrize("failure", [False,True])
def test_adapter_actual_policy_normals_retention_bytes_and_restoration(
    module,monkeypatch,tmp_path,retention,normal,static,failure
):
    adapter = importlib.import_module("scanner_retained_pixels_replay")
    normal_module = importlib.import_module("scanner_normal_scale")
    replay = adapter.static_replay if static else adapter.motion_replay
    original_factory = replay.IncrementalTracker
    factors = module.Factors("grid","pooled","residual")
    output = tmp_path / "output"
    content = b'{"unchanged": true}'
    count = 10 if normal == "observe-10" else 20
    result = SimpleNamespace(normal_neighbours=count,
                             as_dict=lambda: dict(normal_neighbours=count))

    def original(*args,**kwargs):
        assert kwargs.get("normal_neighbours",10) == count
        return result

    monkeypatch.setattr(normal_module,"ORIGINAL_ICP",original)
    monkeypatch.setattr(tracking,"icp_point_to_plane",original)

    def delegated(capture,destination,*args,**kwargs):
        assert capture == tmp_path / "capture" and destination == output
        assert kwargs["diagnostic_source_profile"] == module.SOURCE_PROFILE
        assert kwargs["diagnostic_solver_profile"] == "centered"
        if not static:
            assert args == (320,(1,)) and kwargs["diagnostic_scene"]
            assert kwargs["normals_backend"] == "gpu"
        target,policy = adapter.scanner_source_capacity.settings(
            module.SOURCE_PROFILE,factors,"centered",None)
        assert target == 1800 and policy == module.EXPECTED_POLICY
        value = replay.IncrementalTracker(backend="scipy",policy=policy)
        assert type(value) is module.RetainedPixelTracker
        value._admit(0,cloud(1700),Rigid.identity(),overlap=1.,anchor=True)
        with icp_plane.normal_acceleration(lambda p,k: None):
            assert tracking.icp_point_to_plane([],[],backend="scipy") is result
        output.mkdir()
        (output / "report.json").write_bytes(content)
        if failure:
            raise RuntimeError("injected adapter failure")

    monkeypatch.setattr(replay,"run_static" if static else "run",delegated)
    parameters = dict(retention_profile=retention,normal_profile=normal,
                      solver_profile="centered",static=static,frames=320)
    if failure:
        with pytest.raises(RuntimeError,match="injected adapter failure"):
            adapter.run(tmp_path / "capture",output,factors,**parameters)
    else:
        adapter.run(tmp_path / "capture",output,factors,**parameters)
    assert replay.IncrementalTracker is original_factory
    assert adapter.scanner_source_capacity.settings is adapter.ORIGINAL_SETTINGS
    assert tracking.icp_point_to_plane is original and tracking._sampled is module.ORIGINAL_SAMPLED
    assert not adapter._ACTIVE.get() and not normal_module._ACTIVE.get()
    assert (output / "report.json").read_bytes() == content
    report = json.loads((output / "retention-report.json").read_text())
    assert report["status"] == ("failed" if failure else "completed")
    assert report["retention_profile"] == retention and report["normal_profile"] == normal
    assert report["actual_normal_neighbours"] == count
    assert report["capacity_settings_calls"] == 1
    assert report["tracking_policy"] == module.EXPECTED_POLICY.as_dict()
    assert report["preparation_target_points"] == 1800
    assert report["retention_records"][0]["returned_points"] == 850
    assert report["fit_records"][0]["actual_normal_neighbours"] == count
    assert report["underlying_reports"] == [dict(path=str(output / "report.json"),
        sha256=hashlib.sha256(content).hexdigest())]
    assert not report["quality_qualification_allowed"]


def test_policy_profile_and_nesting_do_not_leak(module,monkeypatch):
    with pytest.raises(ValueError):
        module.RetainedPixelTracker(backend="scipy",policy=DEFAULT_POLICY,
                                   retention_profile="retained-pixel-grid")
    with pytest.raises(ValueError):
        module.RetainedPixelTracker(backend="scipy",policy=module.EXPECTED_POLICY,
                                   retention_profile="unknown")
    value = module.RetainedPixelTracker(backend="scipy",policy=module.EXPECTED_POLICY,
                                       retention_profile="retained-pixel-grid")
    value._inside_retention = True
    with pytest.raises(RuntimeError):
        value._admit(0,cloud(1700),Rigid.identity(),overlap=1.,anchor=True)
    value._inside_retention = False
    value.retention_records = [{}] * 3000
    with pytest.raises(RuntimeError):
        value._admit(0,cloud(1700),Rigid.identity(),overlap=1.,anchor=True)
    value.retention_records = []
    monkeypatch.setattr(tracking,"_sampled",lambda *a,**k: ())
    with pytest.raises(RuntimeError):
        value._admit(0,cloud(1700),Rigid.identity(),overlap=1.,anchor=True)


def test_original_admission_failure_is_preserved_and_hooks_restore(module,monkeypatch):
    value = module.RetainedPixelTracker(backend="scipy",policy=module.EXPECTED_POLICY,
                                       retention_profile="retained-pixel-grid")
    error = RuntimeError("original admission failure")

    def fail(*args,**kwargs):
        raise error

    monkeypatch.setattr(IncrementalTracker,"_update_keyframes",fail)
    with pytest.raises(RuntimeError) as caught:
        value._admit(0,cloud(1700),Rigid.identity(),overlap=1.,anchor=True)
    assert caught.value is error
    assert tracking._sampled is module.ORIGINAL_SAMPLED and not value._inside_retention
    assert value.retention_records[0]["status"] == "failed"
