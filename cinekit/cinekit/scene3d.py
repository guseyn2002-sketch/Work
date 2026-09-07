# -*- coding: utf-8 -*-
"""Blender as a scene renderer, driven headless from Python.

Extruded metal type, product turntables and studio lighting are the elements a
2D pipeline cannot fake. Cycles runs on CPU here; EEVEE works through
surfaceless EGL and is far quicker when raytraced accuracy is not needed.
"""
import json, math, os, shutil, subprocess, tempfile

BLENDER = shutil.which("blender") or "blender"


def available():
    try:
        return subprocess.run([BLENDER, "--version"], capture_output=True,
                              timeout=60).returncode == 0
    except Exception:
        return False


_PRELUDE = r'''
import bpy, math, json, os, sys
CFG = json.loads(os.environ["CINEKIT_CFG"])

def reset():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    for c in (bpy.data.meshes, bpy.data.materials, bpy.data.curves):
        for b in list(c):
            c.remove(b)

def setup(scene_cfg):
    sc = bpy.context.scene
    eng = scene_cfg.get("engine", "CYCLES")
    sc.render.engine = "CYCLES" if eng == "CYCLES" else "BLENDER_EEVEE"
    if sc.render.engine == "CYCLES":
        sc.cycles.device = "CPU"
        sc.cycles.samples = scene_cfg.get("samples", 64)
        sc.cycles.use_denoising = False          # this build ships without OIDN
    else:
        sc.eevee.taa_render_samples = scene_cfg.get("samples", 64)
    sc.render.resolution_x = scene_cfg["width"]
    sc.render.resolution_y = scene_cfg["height"]
    sc.render.resolution_percentage = 100
    sc.render.film_transparent = scene_cfg.get("transparent", True)
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    sc.view_settings.view_transform = scene_cfg.get("view_transform", "Filmic")
    sc.view_settings.look = scene_cfg.get("look", "None")
    return sc

def world_gradient(top, bottom, strength=1.0):
    """A studio environment is what makes metal read as metal."""
    w = bpy.data.worlds.new("env"); bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg  = nt.nodes.new("ShaderNodeBackground")
    grad= nt.nodes.new("ShaderNodeTexGradient"); grad.gradient_type = "EASING"
    mapn= nt.nodes.new("ShaderNodeMapping")
    texc= nt.nodes.new("ShaderNodeTexCoord")
    ramp= nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = tuple(bottom) + (1,)
    ramp.color_ramp.elements[1].color = tuple(top) + (1,)
    mapn.inputs["Rotation"].default_value[1] = math.radians(90)
    nt.links.new(texc.outputs["Generated"], mapn.inputs["Vector"])
    nt.links.new(mapn.outputs["Vector"], grad.inputs["Vector"])
    nt.links.new(grad.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bg.inputs["Color"])
    bg.inputs["Strength"].default_value = strength
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])

def material(name, base=(0.8,0.8,0.85), metallic=1.0, roughness=0.12,
             emission=None, transmission=0.0, coat=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = tuple(base) + (1,)
    b.inputs["Metallic"].default_value = metallic
    b.inputs["Roughness"].default_value = roughness
    for key, val in (("Transmission Weight", transmission), ("Coat Weight", coat)):
        if key in b.inputs:
            b.inputs[key].default_value = val
    if emission:
        if "Emission Color" in b.inputs:
            b.inputs["Emission Color"].default_value = tuple(emission) + (1,)
            b.inputs["Emission Strength"].default_value = 1.0
    return m

def three_point(key=1200, fill=260, rim=800, size=6.0):
    for name, loc, energy, sz in (
        ("key",  (4.2, -4.6,  5.2), key,  size),
        ("fill", (-5.0, -3.0, 1.8), fill, size * 1.6),
        ("rim",  (-1.6,  5.4, 4.2), rim,  size * 0.8)):
        bpy.ops.object.light_add(type="AREA", location=loc)
        lt = bpy.context.object
        lt.data.energy = energy; lt.data.size = sz
        lt.name = name
        d = bpy.data.objects.new(name + "_t", None); bpy.context.collection.objects.link(d)
        c = lt.constraints.new("TRACK_TO"); c.target = d
        c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"

def add_text(cfg):
    bpy.ops.object.text_add()
    ob = bpy.context.object
    ob.data.body = cfg["text"]
    ob.data.extrude = cfg.get("extrude", 0.09)
    ob.data.bevel_depth = cfg.get("bevel", 0.012)
    ob.data.bevel_resolution = 4
    ob.data.align_x = "CENTER"; ob.data.align_y = "CENTER"
    ob.data.size = cfg.get("size", 1.0)
    if cfg.get("font") and os.path.exists(cfg["font"]):
        ob.data.font = bpy.data.fonts.load(cfg["font"])
    ob.rotation_euler = tuple(math.radians(a) for a in cfg.get("rotation", (90, 0, 0)))
    ob.location = tuple(cfg.get("location", (0, 0, 0)))
    m = material("type", base=cfg.get("color", (0.85, 0.86, 0.9)),
                 metallic=cfg.get("metallic", 1.0),
                 roughness=cfg.get("roughness", 0.10),
                 coat=cfg.get("coat", 0.0))
    ob.data.materials.append(m)
    return ob

def add_primitive(cfg):
    kind = cfg.get("kind", "sphere")
    loc = tuple(cfg.get("location", (0, 0, 0)))
    if kind == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=cfg.get("radius", 1.0),
                                             location=loc, segments=64, ring_count=32)
    elif kind == "cube":
        bpy.ops.mesh.primitive_cube_add(size=cfg.get("size", 1.6), location=loc)
    elif kind == "torus":
        bpy.ops.mesh.primitive_torus_add(location=loc,
                                         major_radius=cfg.get("radius", 1.2),
                                         minor_radius=cfg.get("minor", 0.28))
    elif kind == "cylinder":
        bpy.ops.mesh.primitive_cylinder_add(location=loc, vertices=96,
                                            radius=cfg.get("radius", 0.8),
                                            depth=cfg.get("depth", 2.0))
    else:
        bpy.ops.mesh.primitive_plane_add(size=cfg.get("size", 12.0), location=loc)
    ob = bpy.context.object
    if kind != "plane":
        bpy.ops.object.shade_smooth()
    ob.rotation_euler = tuple(math.radians(a) for a in cfg.get("rotation", (0, 0, 0)))
    ob.data.materials.append(material(
        kind, base=cfg.get("color", (0.8, 0.8, 0.85)),
        metallic=cfg.get("metallic", 1.0), roughness=cfg.get("roughness", 0.12),
        transmission=cfg.get("transmission", 0.0), coat=cfg.get("coat", 0.0),
        emission=cfg.get("emission")))
    return ob

def main():
    reset()
    sc = setup(CFG)
    world_gradient(CFG.get("env_top", (0.62, 0.70, 0.85)),
                   CFG.get("env_bottom", (0.04, 0.04, 0.06)),
                   CFG.get("env_strength", 1.1))
    three_point(**CFG.get("lights", {}))
    objs = []
    for o in CFG.get("objects", []):
        objs.append(add_text(o) if o.get("type") == "text" else add_primitive(o))

    cam_cfg = CFG.get("camera", {})
    bpy.ops.object.camera_add(location=tuple(cam_cfg.get("location", (0, -6.2, 0.6))))
    cam = bpy.context.object
    cam.data.lens = cam_cfg.get("lens", 55)
    if cam_cfg.get("dof"):
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = cam_cfg.get("focus", 6.2)
        cam.data.dof.aperture_fstop = cam_cfg.get("fstop", 1.8)
    tgt = bpy.data.objects.new("cam_target", None); bpy.context.collection.objects.link(tgt)
    tgt.location = tuple(cam_cfg.get("target", (0, 0, 0)))
    c = cam.constraints.new("TRACK_TO"); c.target = tgt
    c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    sc.camera = cam

    frames = CFG["frames"]
    orbit = CFG.get("orbit", 0.0)
    rise = CFG.get("rise", 0.0)
    r0 = cam_cfg.get("location", (0, -6.2, 0.6))
    radius = math.hypot(r0[0], r0[1])
    base_a = math.atan2(r0[1], r0[0])
    for k, n in enumerate(frames):
        u = k / max(1, len(frames) - 1)
        a = base_a + math.radians(orbit) * u
        cam.location = (radius * math.cos(a), radius * math.sin(a), r0[2] + rise * u)
        for o, ocfg in zip(objs, CFG.get("objects", [])):
            spin = ocfg.get("spin", 0.0)
            if spin:
                o.rotation_euler[2] = math.radians(spin * u)
        sc.render.filepath = os.path.join(CFG["outdir"], f"{n:05d}.png")
        bpy.ops.render.render(write_still=True)
    print("CINEKIT_3D_DONE")

main()
'''


def render(config, outdir, frames, timeout=3600):
    """Render a scene description to a PNG sequence.

    config keys: width, height, engine, samples, transparent, env_top,
    env_bottom, lights, objects, camera, orbit, rise.
    """
    cfg = dict(config)
    cfg["outdir"] = os.path.abspath(outdir)
    cfg["frames"] = list(frames)
    cfg.setdefault("width", 1080)
    cfg.setdefault("height", 1920)
    os.makedirs(cfg["outdir"], exist_ok=True)

    tmp = tempfile.mkdtemp()
    script = os.path.join(tmp, "scene.py")
    open(script, "w").write(_PRELUDE)
    env = dict(os.environ, CINEKIT_CFG=json.dumps(cfg))
    r = subprocess.run([BLENDER, "-b", "-noaudio", "--python", script],
                       capture_output=True, text=True, env=env, timeout=timeout)
    shutil.rmtree(tmp, ignore_errors=True)
    if "CINEKIT_3D_DONE" not in r.stdout:
        tail = (r.stdout[-1500:] + "\n" + r.stderr[-1500:]).strip()
        raise RuntimeError("blender render failed:\n" + tail)
    return [os.path.join(cfg["outdir"], f"{n:05d}.png") for n in cfg["frames"]]


def chrome_title(text, font_path=None, width=1080, height=1920, frames=range(0, 30),
                 outdir="3d_title", engine="CYCLES", samples=48, orbit=14.0,
                 colour=(0.86, 0.88, 0.93), fill=0.78, lens=50):
    """Extruded metal wordmark on a studio gradient — a 3D logo sting.

    `fill` is the fraction of frame width the word should occupy; the type is
    scaled to it so a long word does not run off the edge.
    """
    aspect = width / float(height)
    dist = 7.6
    # visible width at the subject plane, in Blender units
    view_w = 2.0 * dist * math.tan(math.radians(36.0)) * (36.0 / lens) * max(aspect, 0.4)
    # Blender text size is roughly cap height; a glyph averages ~0.62 of it wide
    size = max(0.16, view_w * fill / max(1, len(text) * 0.62))
    return render(dict(
        width=width, height=height, engine=engine, samples=samples,
        transparent=True, env_top=(0.75, 0.82, 0.95), env_bottom=(0.05, 0.05, 0.08),
        env_strength=1.35,
        lights=dict(key=1400, fill=320, rim=1100, size=7.0),
        objects=[dict(type="text", text=text, font=font_path, size=size,
                      extrude=size * 0.13, bevel=size * 0.016, color=colour,
                      metallic=1.0, roughness=0.08, rotation=(90, 0, 0))],
        camera=dict(location=(0, -dist, 0.0), lens=lens, target=(0, 0, 0),
                    dof=True, focus=dist, fstop=2.8),
        orbit=orbit, rise=0.12,
    ), outdir, frames)
