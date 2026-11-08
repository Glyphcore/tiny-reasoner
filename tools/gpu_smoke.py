"""One-line GPU check. WSL2 CUDA needs /usr/lib/wsl/lib on LD_LIBRARY_PATH; re-exec once with it set if torch cannot see the GPU."""
import os, sys

WSL = "/usr/lib/wsl/lib"


def main() -> int:
    import torch
    if not torch.cuda.is_available() and WSL not in os.environ.get("LD_LIBRARY_PATH", "") and os.path.isdir(WSL):
        os.environ["LD_LIBRARY_PATH"] = WSL + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        os.execv(sys.executable, [sys.executable] + sys.argv)
    if not torch.cuda.is_available():
        print("FAIL cuda unavailable (torch %s, cuda build %s)" % (torch.__version__, torch.version.cuda))
        return 1
    try:
        a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
        ok = bool(torch.isfinite(a @ a).all())
    except Exception as e:  # noqa: BLE001
        ok = False
        print("matmul error:", str(e)[:120], file=sys.stderr)
    free, _ = torch.cuda.mem_get_info()
    print("%s | fp16 matmul %s | free VRAM %.1f GB" % (torch.cuda.get_device_name(0), "ok" if ok else "fail", free / 2**30))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
