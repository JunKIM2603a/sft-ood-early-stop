from pathlib import Path


def test_p2_aio_setup_has_non_root_fallback():
    script = Path("scripts/setup_deepspeed_aio.sh").read_text(
        encoding="utf-8"
    )

    assert "apt-get download libaio-dev" in script
    assert "dpkg-deb -x" in script
    assert "artifacts/env/deepspeed_aio.env" in script


def test_p2_preflight_points_to_aio_setup():
    script = Path("scripts/data/preflight_modelscale_p2.py").read_text(
        encoding="utf-8"
    )

    assert "setup_deepspeed_aio.sh" in script
    assert "apply_env_file(AIO_ENV)" in script
