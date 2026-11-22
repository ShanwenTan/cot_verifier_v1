"""Blender worker. Add cameras only after running the exact training target."""
import bpy
from mathutils import Vector
import json
import math
import os
import sys
from pathlib import Path

root = Path.cwd()
config = json.loads(Path(os.environ.get("SFT_CONFIG_PATH", root / "config.json")).read_text())
metrics = {"blender": bpy.app.version_string, "script_completed": False}
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

source = Path(os.environ.get("SFT_ASSET_PATH", root / "asset.py"))
exec(compile(source.read_bytes(), str(source), "exec"),
     {"__name__": "__main__", "__file__": str(source)})
metrics["script_completed"] = True
scene = bpy.context.scene
graph = bpy.context.evaluated_depsgraph_get()
geometry = []
for obj in scene.objects:
    if obj.type not in {"MESH", "CURVE", "FONT", "SURFACE"}:
        continue
    evaluated = obj.evaluated_get(graph)
    mesh = evaluated.to_mesh()
    try:
        if not mesh or not len(mesh.polygons):
            continue
        points = [evaluated.matrix_world @ Vector(corner) for corner in evaluated.bound_box]
        if not all(math.isfinite(c) for v in mesh.vertices for c in v.co):
            metrics["nonfinite_vertices"] = True
        geometry.append((obj, points, len(mesh.vertices), len(mesh.polygons)))
    finally:
        evaluated.to_mesh_clear()
metrics.update(objects=len(geometry), vertices=sum(g[2] for g in geometry),
               faces=sum(g[3] for g in geometry), object_names=[g[0].name for g in geometry],
               camera_hidden=[g[0].name for g in geometry if not g[0].visible_camera or g[0].hide_render],
               views=[])
(root / "metrics.json").write_text(json.dumps(metrics, indent=2))
if not geometry or metrics.get("nonfinite_vertices"):
    sys.exit(0)
points = [v for _, corners, *_ in geometry for v in corners]
lo = Vector(tuple(min(p[i] for p in points) for i in range(3)))
hi = Vector(tuple(max(p[i] for p in points) for i in range(3)))
dims = hi - lo
scale = max(dims)
if not all(math.isfinite(x) for x in dims) or scale <= 1e-9:
    metrics["invalid_bounds"] = True
    (root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sys.exit(0)

metrics["dimensions_xyz"] = [round(x, 5) for x in dims]
center = (lo + hi) * .5
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
if os.environ.get("SFT_CUDA") == "1":
    try:
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "CUDA"
        preferences.refresh_devices()
        devices = [device for device in preferences.devices if device.type == "CUDA"]
        if len(devices) != 1:
            raise RuntimeError(f"expected one isolated GPU, got {len(devices)}")
        for device in preferences.devices:
            device.use = device.type == "CUDA"
        scene.cycles.device = "GPU"
        metrics["render_device"] = devices[0].name
    except Exception as error:
        print(f"SFT_GPU_UNAVAILABLE: {error}", file=sys.stderr, flush=True)
        import ctypes
        try:
            cuda = ctypes.CDLL("libcuda.so.1")
            count = ctypes.c_int(-1)
            err = cuda.cuInit(0)
            rc = cuda.cuDeviceGetCount(ctypes.byref(count))
            print(f"SFT_DEBUG_CUDA init={err} count={count.value} rc={rc} dev={os.listdir('/dev')}", flush=True)
            paths = "/proc/driver/nvidia/gpus", "/sys/bus/pci/devices"
            for path in paths:
                if os.path.isdir(path):
                    print("SFT_DEBUG_PATH", path, len(os.listdir(path)), flush=True)
            for path in ["/dev/nvidiactl", "/dev/nvidia-uvm"] + ["/dev/" + x for x in os.listdir("/dev") if x.startswith("nvidia") and x[6:].isdigit()]:
                try:
                    fd = os.open(path, os.O_RDWR)
                    os.close(fd)
                    print("SFT_DEBUG_DEVICE", path, "ok", flush=True)
                except OSError as exc:
                    print("SFT_DEBUG_DEVICE", path, exc, flush=True)
        except Exception as exc:
            print("SFT_DEBUG_CUDA", exc, flush=True)

        raise
else:
    metrics["render_device"] = "CPU"
scene.cycles.samples = 16
scene.cycles.use_denoising = True
scene.render.threads_mode = "FIXED"
scene.render.threads = config["threads"]
scene.render.resolution_x = scene.render.resolution_y = config["size"]
scene.render.resolution_percentage = 100
scene.render.film_transparent = True
scene.render.image_settings.file_format = "PNG"
scene.render.image_settings.color_mode = "RGBA"
scene.view_settings.view_transform = "AgX"

world = bpy.data.worlds.new("Audit World")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (.69, .76, .85, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = .52
scene.world = world

def face(obj, location):
    obj.rotation_euler = (location - obj.location).to_track_quat("-Z", "Y").to_euler()

for name, direction, power in (("Key",(-2,-2,3),155),("Fill",(2,1,2),80)):
    light = bpy.data.lights.new(name, "AREA")
    light.energy = power * scale * scale
    light.shape = "DISK"
    light.size = 1.6 * scale
    obj = bpy.data.objects.new(name, light)
    scene.collection.objects.link(obj)
    obj.location = center + scale * Vector(direction)
    face(obj, center)

camera = bpy.data.cameras.new("Audit camera")
camera.type = "ORTHO"
camera.clip_start = scale * .002
camera.clip_end = scale * 40
view = bpy.data.objects.new("Audit camera", camera)
scene.collection.objects.link(view)
scene.camera = view
for i, direction in enumerate(config["views"], 1):
    view.location = center + scale * Vector(direction).normalized() * 3
    face(view, center)
    bpy.context.view_layer.update()
    projected = [view.matrix_world.inverted() @ p for p in points]
    camera.ortho_scale = 1.22 * max(max(p[j] for p in projected)-min(p[j] for p in projected) for j in (0,1))
    name = f"view-{i}.png"
    scene.render.filepath = str(root / name)
    bpy.ops.render.render(write_still=True)
    metrics["views"].append(name)
(root / "metrics.json").write_text(json.dumps(metrics, indent=2))
