"""Contract/integration tests for the per-model studio API (image gen + GPU mocked).

Mounts studio.router on a fresh FastAPI app and drives it with TestClient. The network-lane
reference generator is replaced with a fast local image; GPU dispatch is stubbed. Covers model
CRUD + persistence, staged-generation gating, the job->model contract, base-guard, and download.

Run:  python webapp/test_studio_api.py     (or via pytest)
"""
import io
import os
import tempfile
import time

os.environ["HY3D_OUTPUT_DIR"] = tempfile.mkdtemp(prefix="studio_test_")

from fastapi import FastAPI                       # noqa: E402
from fastapi.testclient import TestClient         # noqa: E402
from PIL import Image                             # noqa: E402

import webapp.reference_views as rv               # noqa: E402
from webapp import studio                         # noqa: E402


def _png(color=(180, 80, 80)):
    b = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(b, format="PNG")
    return b.getvalue()


def _fake_gen(view, seed, deps, edit=None, size=(64, 64)):
    return Image.new("RGB", (64, 64), (10, 120, 120))


rv.generate_view = _fake_gen  # no real gpt-image calls


def _client():
    app = FastAPI()
    app.include_router(studio.router)

    @app.get("/api/jobs/{jid}")
    def _job(jid):  # mirrors server.job_status delegation
        return studio.public_job(jid)

    return TestClient(app)


def _create(c, name="M"):
    r = c.post("/api/models", data={"name": name},
               files={"seed_image": ("s.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _poll(c, jid, timeout=15):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] == "completed":
            assert j["model"], "completed job must embed the model"
            return j["model"]
        if j["status"] == "failed":
            raise AssertionError("job failed: " + str(j.get("error")))
        time.sleep(0.05)
    raise AssertionError("job did not complete in time")


def test_crud_and_persistence():
    c = _client()
    mid = _create(c, "Robo")
    m = c.get(f"/api/models/{mid}").json()
    assert m["id"] == mid and m["name"] == "Robo" and m["seedImageUrl"]
    assert c.patch(f"/api/models/{mid}", json={"name": "Robo2"}).json()["name"] == "Robo2"
    assert mid in [x["id"] for x in c.get("/api/models").json()]
    # disk-backed: a brand-new client (fresh app) still lists it -> survives "restart"
    assert mid in [x["id"] for x in _client().get("/api/models").json()]
    assert c.delete(f"/api/models/{mid}").status_code == 204
    assert c.get(f"/api/models/{mid}").status_code == 404


def test_reference_gating_and_generate():
    c = _client()
    mid = _create(c, "Gate")
    # cardinal before front approved -> 409
    assert c.post(f"/api/models/{mid}/references/left/generate", json={}).status_code == 409
    # generate front (mocked network lane) and poll to completion
    r = c.post(f"/api/models/{mid}/references/front/generate", json={})
    assert r.status_code == 200 and isinstance(r.json()["progress"], (int, float))
    model = _poll(c, r.json()["id"])
    assert model["references"]["front"]["status"] == "pending"
    assert model["references"]["front"]["url"] and model["references"]["front"]["source"] == "generated"
    # approve an empty view -> 400
    assert c.post(f"/api/models/{mid}/references/left/approve").status_code == 400
    # approve front -> a view depending only on front (back) is now allowed
    assert c.post(f"/api/models/{mid}/references/front/approve").json()["references"]["front"]["status"] == "approved"
    assert c.post(f"/api/models/{mid}/references/back/generate", json={}).status_code == 200
    # left also needs back + top approved, so it is still gated
    assert c.post(f"/api/models/{mid}/references/left/generate", json={}).status_code == 409


def test_upload_autoapprove_and_base_guard():
    c = _client()
    mid = _create(c, "Up")
    m = c.post(f"/api/models/{mid}/references/front/upload",
               files={"image": ("f.png", _png(), "image/png")}).json()
    assert m["references"]["front"]["status"] == "approved"
    assert m["references"]["front"]["source"] == "uploaded"
    # base requires all 10 approved
    assert c.post(f"/api/models/{mid}/texture/base", json={}).status_code == 400


def test_base_dispatch_when_all_approved():
    c = _client()
    mid = _create(c, "Base")
    for v in rv.ALL_VIEWS:
        c.post(f"/api/models/{mid}/references/{v}/upload", files={"image": ("f.png", _png(), "image/png")})
    calls = []
    orig = studio.submit_gpu
    studio.submit_gpu = lambda kind, sjid: calls.append((kind, sjid))
    try:
        r = c.post(f"/api/models/{mid}/texture/base", json={})
        assert r.status_code == 200, r.text
        assert calls and calls[0][0] == "studio_base"
        assert c.get(f"/api/models/{mid}").json()["textureStage"] == "base-running"
    finally:
        studio.submit_gpu = orig


def test_reface_and_download_guards():
    c = _client()
    mid = _create(c, "Dl")
    # reface before any textured mesh -> 400
    assert c.post(f"/api/models/{mid}/texture/reface/front", json={}).status_code == 400
    # bad download format -> 400; no mesh yet -> glb 404
    assert c.get(f"/api/models/{mid}/download/stl").status_code == 400
    assert c.get(f"/api/models/{mid}/download/glb").status_code == 404
    # unknown view -> 400
    assert c.post(f"/api/models/{mid}/references/sideways/generate", json={}).status_code == 400


def test_replace_seed_updates_seed_not_front():
    c = _client()
    mid = _create(c, "Seed")
    m1 = c.get(f"/api/models/{mid}").json()
    assert m1["seedImageUrl"]
    # replace the seed -> updates the seed (new cache-busted URL), NOT the front reference
    m2 = c.post(f"/api/models/{mid}/seed", files={"image": ("s2.png", _png((10, 200, 10)), "image/png")}).json()
    assert m2["seedImageUrl"] and m2["seedImageUrl"] != m1["seedImageUrl"]
    assert m2["references"]["front"]["status"] == "empty"
    assert m2["references"]["front"]["url"] is None
    # 404 on unknown model
    assert c.post("/api/models/00000000-0000-0000-0000-000000000000/seed",
                  files={"image": ("s.png", _png(), "image/png")}).status_code == 404


def test_mesh_endpoint():
    c = _client()
    mid = _create(c, "Mesh")
    # unknown view -> 400
    assert c.post(f"/api/models/{mid}/mesh", json={"source_view": "sideways"}).status_code == 400
    # valid view but no reference image yet -> 400
    assert c.post(f"/api/models/{mid}/mesh", json={"source_view": "front"}).status_code == 400
    # give front an image, then /mesh dispatches a GPU 'studio_mesh' job
    c.post(f"/api/models/{mid}/references/front/upload", files={"image": ("f.png", _png(), "image/png")})
    calls = []
    orig = studio.submit_gpu
    studio.submit_gpu = lambda kind, sjid: calls.append((kind, sjid))
    try:
        r = c.post(f"/api/models/{mid}/mesh", json={"source_view": "front"})
        assert r.status_code == 200, r.text
        assert calls and calls[0][0] == "studio_mesh"
    finally:
        studio.submit_gpu = orig


def test_masked_edit_reference():
    c = _client()
    mid = _create(c, "MaskEdit")
    png = _png()
    # no image to edit yet -> 400
    r0 = c.post(f"/api/models/{mid}/references/front/edit",
                files={"mask": ("m.png", png, "image/png")}, data={"edit_prompt": "x"})
    assert r0.status_code == 400
    # generate a front (mocked) so there is an image to edit
    jid = c.post(f"/api/models/{mid}/references/front/generate", json={}).json()["id"]
    _poll(c, jid)
    # stub the masked inpaint (no real gpt-image call)
    rv.edit_view_masked = lambda cur, mask, edit, size=(64, 64): Image.new("RGB", (64, 64), (40, 40, 200))
    r = c.post(f"/api/models/{mid}/references/front/edit",
               files={"mask": ("m.png", png, "image/png")}, data={"edit_prompt": "make it blue"})
    assert r.status_code == 200 and isinstance(r.json()["progress"], (int, float))
    model = _poll(c, r.json()["id"])
    assert model["references"]["front"]["status"] == "pending"
    assert model["references"]["front"]["url"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("all studio_api tests passed")
