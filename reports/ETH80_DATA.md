# ETH-80 data manifest

- **Local corpus:** `/home/user/ETH80` (kept outside the Melissia Git repository; not added to Git).
- **Retrieved:** 2026-10-16 from the public GitHub mirror [`chenchkx/ETH-80`](https://github.com/chenchkx/ETH-80), commit `9243466b9ca5f15765c88fdaf17c4f10c15a8574`.
- **Verified layout:** 8 categories × 10 object instances × 41 RGB views = **3,280 photos**. The mirror also contains one `-map.png` mask per photo, for **6,560 PNG files total**. Each category has 410 photos and 410 masks; RGB images are 256×256. All 6,560 PNG files passed Pillow's image verification (0 corrupt files).
- **Size:** approximately 631 MB.
- **Harness handling:** recursive photo enumeration excludes `-map.png`; images were read as full frames. No image-splitting code or preprocessing was added.
- **Provenance caveat:** this is a GitHub mirror, not a current official ETH Zurich download endpoint. The mirror has no `LICENSE`/`COPYING` file. Its README cites Leibe & Schiele, CVPR 2003. The data is therefore kept external and is not redistributed or committed; verify the original dataset terms before redistributing it.
