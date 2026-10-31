# Img-2-3D

Android app that runs the **original TripoSR** (VAST-AI-Research code, `stabilityai/TripoSR` weights) fully on-device: gallery image → background removal → TripoSR network → density field → marching cubes → vertex colors → GLB, with an interactive viewer, history, export and sharing. No cloud inference.

## How the port works

| TripoSR (Python) | On Android |
|---|---|
| `rembg.remove` (u2net) | same `u2net.onnx` on ONNX Runtime, rembg pre/post-processing ported (`TripoSR.removeBackground`) |
| `resize_foreground`, gray composite, `ImagePreprocessor` | `ImageOps` (antialiased bilinear identical to `F.interpolate(antialias=True)`) |
| `TSR.forward` (DINO ViT-B/16 tokenizer, transformer backbone, triplane upsampler) | `triposr_encoder.onnx`, traced from the original modules, weights unchanged (fp32) |
| `renderer.query_triplane` + `NeRFMLP` (grid_sample + MLP, density/color activations) | `triposr_field.onnx`, queried in 65k-point chunks |
| `MarchingCubeHelper` / torchmcubes | `MarchingCubes.java`, line-by-line port with the same tables, axis swap and scaling |
| trimesh export | `Mesh.writeGlb` (Y-up like TripoSR's gradio app, vertex colors) |

The pipeline is plain Java (`:core`) so the **same code** runs on the phone and in CI on the desktop JVM.

## Verification

`.github/workflows/models.yml` (run it once from the Actions tab):
1. exports both graphs and asserts ONNX == PyTorch (max abs error on scene codes, relative error on densities),
2. runs the app's Java pipeline on the reference tensor and requires chamfer < 0.004 vs the mesh produced by the original `TSR.extract_mesh`,
3. runs the full pipeline from a PNG (Java preprocessing + u2net) with chamfer < 0.03, logs time and peak RSS,
4. validates the GLBs with trimesh, uploads them as artifacts, and publishes the model files to the `models-v1` release.

`.github/workflows/android.yml` runs core unit tests (watertight/oriented marching cubes, GLB structure, torch-equivalent resize) and builds the APKs.

**On your phone** (after installing the model):
```
gradle :app:connectedDebugAndroidTest -Pandroid.testInstrumentationRunnerArguments.class=com.smitnk.img23d.OnDevicePipelineTest
adb logcat -s Img23DBench
```
prints faces, timings per stage and peak PSS. The app's result screen shows the same numbers for every model you generate.

## Installing the model

The weights are ~1.9 GB and can't ship inside an APK. Either tap **Download model** in the app (one-time, from this repo's `models-v1` release, SHA-256 verified), or push them:
```
adb push model_info.json triposr_encoder.onnx triposr_encoder.onnx.data triposr_field.onnx u2net.onnx \
  /sdcard/Android/data/com.smitnk.img23d/files/models/
```
then tap **verify** on the model screen.

## Build

```
gradle :core:test :app:assembleRelease
```
Requires JDK 17, Android SDK 35, Gradle 8.11.

## Notes for the Dimensity 7200 / 8 GB target
- CPU inference via ONNX Runtime (fp32, up to 6 threads). The encoder session (~1.7 GB) is created only for its stage and closed before meshing; the field session is small.
- Mesh resolution 128³/192³/256³ is selectable; 256³ needs a 67 MB density volume plus mesh buffers (`largeHeap`).
- One job at a time, no background service; the screen stays on while generating.
- GPU/NNAPI acceleration is not enabled: not yet validated for numerical equivalence with these graphs.

Licenses: TripoSR MIT (`LICENSE-TripoSR`), torchmcubes MIT (`core/LICENSE-torchmcubes`).
