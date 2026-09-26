# DeepSpeed async_io prerequisite for P2 NVMe offload

DeepSpeed DeepNVMe requires the `async_io` operator. That operator depends on
the Linux `libaio` development headers and shared library.

Official DeepSpeed guidance for Ubuntu is:

```bash
sudo apt-get install -y libaio-dev
```

The project helper supports two paths:

1. if passwordless sudo is available, install `libaio-dev` system-wide;
2. otherwise download the distro `libaio-dev` and runtime `.deb` packages,
   extract them under `artifacts/local/libaio`, and write
   `artifacts/env/deepspeed_aio.env` so P2 can use them without root.

Run:

```bash
bash scripts/setup_deepspeed_aio.sh
python scripts/data/preflight_modelscale_p2.py
```

A successful setup must end with `async_io compatible=True`.
Do not start the 7B smoke until the P2 preflight passes.
