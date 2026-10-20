import math
import os
import time

import bpy
import numpy as np
from bpy_extras.io_utils import ExportHelper, ImportHelper
from mathutils import Vector

from . import extract, i18n, preview, training
from .core import analytic, axes, baseline
from .core.gaussians import GaussianSet
from .io import ply, spz
from .props import AVAILABLE_METHODS, TRAINED_METHODS

tr = i18n.tr
CTX = i18n.CTX


def spaced(n):
    return f"{int(n):,}".replace(",", " ")


def selected_sources(context):
    return [obj for obj in context.selected_objects
            if obj.type == 'MESH' and not preview.is_gaussian_object(obj)]


def source_objects(gs_obj):
    names = gs_obj.get("mesh2gs_sources", "")
    return [bpy.data.objects[n] for n in names.split(";") if n in bpy.data.objects]


def budget_of(settings, mesh):
    """The Gaussian count the settings ask for on `mesh` (a count, or a density times the area)."""
    if settings.budget_mode == 'COUNT':
        return settings.budget
    return max(1, round(settings.density * analytic.surface_area(mesh.vertices, mesh.triangles)))


def convert_scene(context, objects, selected_faces_only=False, method=None):
    """Run M1 or M0 (`method`, default: the selected one) on `objects` (Object Mode);
    returns (GaussianSet, seconds, warnings)."""
    s = context.scene.mesh2gs
    method = method or s.method
    started = time.perf_counter()
    lit = s.color_mode == 'LIT'
    mesh = extract.scene_mesh(objects, context.evaluated_depsgraph_get(),
                              selected_faces_only=selected_faces_only,
                              bake_resolution=int(s.bake_resolution) if (s.bake_procedural or lit) else 0,
                              lit=lit, lit_samples=s.lit_samples, emission=s.use_emission)
    if method == 'M1':
        gaussians = analytic.convert(mesh.vertices, mesh.triangles, mesh.colors, budget_of(s, mesh),
                                     overlap=s.overlap, seed=s.seed, sampling=s.sampling,
                                     edge_aware=s.edge_aware, sharp_angle=math.degrees(s.sharp_angle))
    elif method == 'M0':
        vertices, triangles = baseline.weld(mesh.vertices, mesh.triangles,
                                            mesh.corner_normals, mesh.corner_uvs)
        # KIRI reads the texture only at triangle corners, nearest texel.
        gaussians = baseline.convert(vertices, triangles, mesh.colors.corner_colors(mode="kiri"),
                                     int(s.per_triangle))
    else:
        raise ValueError(f"method {method} is trained (TrainJob), not converted here")
    return gaussians, time.perf_counter() - started, mesh.warnings


def place_result(context, sources, gaussians, method, seconds, select=True):
    """The result as a Gaussian object next to its sources; with `select`, the sources are
    deselected (and hidden, if set) and the result becomes active."""
    s = context.scene.mesh2gs
    base = sources[0].name if len(sources) == 1 else "Mesh2GS"
    obj = preview.write_object(f"{base}.{method}", gaussians, method, seconds, s.preview_limit)
    obj["mesh2gs_sources"] = ";".join(o.name for o in sources)
    if select:
        for src in sources:
            src.select_set(False)
            if s.hide_source:
                src.hide_set(True)
        obj.hide_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
    return obj


def _redraw(context):
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {'VIEW_3D', 'PREFERENCES'}:
                area.tag_redraw()


def _status(context, text):
    if context.workspace is not None:
        context.workspace.status_text_set(text)


class MESH2GS_OT_convert(bpy.types.Operator):
    bl_idname = "mesh2gs.convert"
    bl_translation_context = CTX
    bl_label = "Convert to Gaussians"
    bl_description = "Build Gaussians on the selected mesh objects with the chosen method"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        if not selected_sources(context):
            cls.poll_message_set(tr("Select at least one mesh object"))
            return False
        method = context.scene.mesh2gs.method
        if method not in AVAILABLE_METHODS:
            cls.poll_message_set(tr("This method comes in a later version"))
            return False
        if training.ACTIVE is not None and training.ACTIVE.running:
            cls.poll_message_set(tr("Training is running; wait for it or cancel it"))
            return False
        if method in TRAINED_METHODS and not training.python_path(context):
            cls.poll_message_set(tr("Set a Python with PyTorch and gsplat in the add-on preferences"))
            return False
        return True

    def invoke(self, context, event):
        wm = context.window_manager
        if bpy.app.version >= (4, 1, 0):
            return wm.invoke_props_dialog(self, width=360, title="Mesh → Gaussian Splatting",
                                          confirm_text=tr("Convert"))
        return wm.invoke_props_dialog(self, width=360)

    def draw(self, context):
        from .ui import draw_settings
        draw_settings(self.layout, context)

    def execute(self, context):
        s = context.scene.mesh2gs
        if s.method in TRAINED_METHODS:
            return self.start_training(context)
        sources = selected_sources(context)
        # Baking and a synced mesh both need Object Mode; the edit session comes back after.
        was_editing = context.mode == 'EDIT_MESH'
        if was_editing:
            bpy.ops.object.mode_set(mode='OBJECT')
        try:
            gaussians, seconds, warnings = convert_scene(
                context, sources, selected_faces_only=s.selected_faces_only and was_editing)
        except ValueError as err:
            self.report({'ERROR'}, tr(str(err)))
            return {'CANCELLED'}
        finally:
            if was_editing:
                bpy.ops.object.mode_set(mode='EDIT')

        place_result(context, sources, gaussians, s.method, seconds, select=not was_editing)
        for warning in warnings:
            self.report({'WARNING'}, warning)
        self.report({'INFO'}, tr("{}: {} Gaussians in {:.2f} s").format(s.method, spaced(len(gaussians)), seconds))
        return {'FINISHED'}

    # -- M2 / M3: views and training advance step by step under a modal handler -------
    _timer = None

    def start_training(self, context):
        s = context.scene.mesh2gs
        if context.mode != 'OBJECT':
            self.report({'ERROR'}, tr("M2 and M3 work in Object Mode"))
            return {'CANCELLED'}
        sources = selected_sources(context)
        started = time.perf_counter()
        warnings = []
        try:
            if s.method == 'M3':
                init, _, warnings = convert_scene(context, sources, method='M1')
                budget, bbox = len(init), None
            else:
                init = None
                mesh = extract.scene_mesh(sources, context.evaluated_depsgraph_get())  # geometry only
                budget = budget_of(s, mesh)
                bbox = (*mesh.vertices.min(axis=0), *mesh.vertices.max(axis=0))
            job = training.TrainJob(
                context, s.method, sources, budget,
                s.m3_iterations if s.method == 'M3' else s.m2_iterations,
                init=init, bbox=bbox, views=s.train_views, size=int(s.train_size),
                lit=s.color_mode == 'LIT', samples=s.lit_samples, emission=s.use_emission)
        except (ValueError, RuntimeError, OSError) as err:
            self.report({'ERROR'}, tr(str(err)))
            return {'CANCELLED'}
        job.started = started  # the reported time includes M1 and the views
        for warning in warnings:
            self.report({'WARNING'}, warning)
        training.ACTIVE = job
        if bpy.app.background:
            job.run()
            return self.finish_training(context)
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.05, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        job = training.ACTIVE
        if job is None:
            return self.finish_training(context)
        if event.type == 'ESC' and job.stage == "VIEWS":
            job.cancel()
        elif event.type == 'TIMER':
            job.step()
        if not job.running:
            return self.finish_training(context)
        hint = tr("Esc cancels") if job.stage == "VIEWS" else tr("cancel on the Mesh2GS panel")
        _status(context, f"Mesh2GS · {job.text} · {hint}")
        _redraw(context)
        # Views render in this Blender: hold the UI meanwhile. Training runs in another
        # process: let the user work.
        return {'RUNNING_MODAL'} if job.stage == "VIEWS" else {'PASS_THROUGH'}

    def cancel(self, context):  # Blender drops the handler (file loaded, window closed)
        if training.ACTIVE is not None:
            training.ACTIVE.cancel()
        self.finish_training(context)

    def finish_training(self, context):
        job, training.ACTIVE = training.ACTIVE, None
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        if not bpy.app.background:
            _status(context, None)
        if job is None:
            return {'CANCELLED'}
        try:
            if job.stage == "DONE":
                gaussians = job.result()
                sources = [bpy.data.objects[n] for n in job.source_names if n in bpy.data.objects]
                place_result(context, sources, gaussians, job.method, job.seconds,
                             select=context.mode == 'OBJECT')  # the user may have moved on
                self.report({'INFO'}, tr("{}: {} Gaussians in {:.1f} s").format(
                    job.method, spaced(len(gaussians)), job.seconds))
                return {'FINISHED'}
            if job.stage == "CANCELLED":
                self.report({'WARNING'}, tr("{}: cancelled").format(job.method))
            else:
                self.report({'ERROR'}, f"{job.method}: {tr(job.error)}")
            return {'CANCELLED'}
        finally:
            job.cleanup()
            if not bpy.app.background:
                _redraw(context)


SCENE_LIGHT_WORLD = "Mesh2GS scene light"


def bounds_center(objects):
    """World-space center of the objects' bounding boxes."""
    corners = np.array([obj.matrix_world @ Vector(c) for obj in objects for c in obj.bound_box])
    return (corners.min(axis=0) + corners.max(axis=0)) / 2


class MESH2GS_OT_scene_light(bpy.types.Operator):
    bl_idname = "mesh2gs.scene_light"
    bl_translation_context = CTX
    bl_label = "Light from splat scene"
    bl_description = ("Render a 360° panorama of the chosen splat scene from the center of the selected "
                      "objects and make it the World; the Scene light color mode then bakes the scene's own "
                      "light into the object. Needs a Python with PyTorch and gsplat (NVIDIA)")
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        scene_obj = context.scene.mesh2gs.light_scene
        if scene_obj is None or not preview.is_gaussian_object(scene_obj):
            cls.poll_message_set(tr("Choose a splat scene (an imported PLY/SPZ or a conversion result)"))
            return False
        if not selected_sources(context):
            cls.poll_message_set(tr("Select the objects that go into the scene"))
            return False
        if not training.python_path(context):
            cls.poll_message_set(tr("Set a Python with PyTorch and gsplat in the add-on preferences"))
            return False
        return True

    def execute(self, context):
        s = context.scene.mesh2gs
        started = time.perf_counter()
        eye = bounds_center(selected_sources(context))
        try:
            path = training.scene_panorama(context, preview.read_object(s.light_scene), eye)
        except (RuntimeError, OSError) as err:
            self.report({'ERROR'}, tr(str(err)))
            return {'CANCELLED'}
        try:
            image = bpy.data.images.load(path)
            image.pack()  # the file is temporary
        finally:
            training.remove_panorama(path)
        image.name = f"{s.light_scene.name} panorama"
        world = bpy.data.worlds.get(SCENE_LIGHT_WORLD) or bpy.data.worlds.new(SCENE_LIGHT_WORLD)
        world.use_nodes = True
        nodes, links = world.node_tree.nodes, world.node_tree.links
        texture = next((n for n in nodes if n.type == 'TEX_ENVIRONMENT'), None) or nodes.new('ShaderNodeTexEnvironment')
        old, texture.image = texture.image, image
        if old is not None and old.users == 0:
            bpy.data.images.remove(old)
        background = next((n for n in nodes if n.type == 'BACKGROUND'), None) or nodes.new('ShaderNodeBackground')
        background.inputs["Strength"].default_value = 1.0  # the splats hold the scene's own brightness
        links.new(texture.outputs["Color"], background.inputs["Color"])
        output = next((n for n in nodes if n.type == 'OUTPUT_WORLD'), None) or nodes.new('ShaderNodeOutputWorld')
        links.new(background.outputs["Background"], output.inputs["Surface"])
        context.scene.world = world
        s.color_mode = 'LIT'
        self.report({'INFO'}, tr("World: panorama of “{}” from ({:.2f}, {:.2f}, {:.2f}) in {:.1f} s; "
                                 "color mode set to Scene light").format(
            s.light_scene.name, *eye, time.perf_counter() - started))
        return {'FINISHED'}


class MESH2GS_OT_cancel_training(bpy.types.Operator):
    bl_idname = "mesh2gs.cancel_training"
    bl_translation_context = CTX
    bl_label = "Cancel training"
    bl_description = "Stop the M2/M3 training; no result will appear"

    @classmethod
    def poll(cls, context):
        return training.ACTIVE is not None and training.ACTIVE.running

    def execute(self, context):
        training.ACTIVE.cancel()
        return {'FINISHED'}


def _addon_prefs(context):
    addon = context.preferences.addons.get(__package__)
    return getattr(addon, "preferences", None)


def _watch_install():
    """Timer while the components install: redraw the preferences, then use the result."""
    from . import setup_env
    job = setup_env.INSTALLER
    _redraw(bpy.context)
    if job is None or job.running:
        return 1.0
    if job.stage == "DONE":
        prefs = _addon_prefs(bpy.context)
        if prefs is not None:
            prefs.python = job.python
        training.CHECKED[job.python] = True, job.result
    return None


class MESH2GS_OT_install_env(bpy.types.Operator):
    bl_idname = "mesh2gs.install_env"
    bl_translation_context = CTX
    bl_label = "Install components (≈5 GB)"
    bl_description = ("Download a Python 3.10 of its own, PyTorch with CUDA and gsplat into the chosen folder "
                      "and use them for M2, M3 and the splat-scene light. Takes several minutes; "
                      "Blender stays usable")

    @classmethod
    def poll(cls, context):
        from . import setup_env
        if setup_env.INSTALLER is not None and setup_env.INSTALLER.running:
            cls.poll_message_set(tr("Installation is running"))
            return False
        return setup_env.supported()

    def execute(self, context):
        from . import setup_env
        from .props import setup_env_folder
        prefs = _addon_prefs(context)
        folder = setup_env_folder(prefs)
        job = setup_env.EnvInstaller(folder)
        try:
            job.start()
        except (RuntimeError, OSError) as err:
            self.report({'ERROR'}, str(err))
            return {'CANCELLED'}
        setup_env.INSTALLER = job
        bpy.app.timers.register(_watch_install, first_interval=1.0)
        self.report({'INFO'}, tr("Installing into {}").format(folder))
        return {'FINISHED'}


class MESH2GS_OT_cancel_install(bpy.types.Operator):
    bl_idname = "mesh2gs.cancel_install"
    bl_translation_context = CTX
    bl_label = "Cancel installation"
    bl_description = "Stop downloading and installing the components; what is already there stays"

    @classmethod
    def poll(cls, context):
        from . import setup_env
        return setup_env.INSTALLER is not None and setup_env.INSTALLER.running

    def execute(self, context):
        from . import setup_env
        setup_env.INSTALLER.cancel()
        return {'FINISHED'}


class MESH2GS_OT_copy_setup(bpy.types.Operator):
    bl_idname = "mesh2gs.copy_setup"
    bl_translation_context = CTX
    bl_label = "Copy commands"
    bl_description = "Copy to the clipboard the commands that make a Python with PyTorch and gsplat (mesh2gs-venv)"

    def execute(self, context):
        commands = training.setup_commands()
        context.window_manager.clipboard = "\n".join(commands) + "\n"
        self.report({'INFO'}, tr("{} commands copied: paste them into a terminal").format(len(commands)))
        return {'FINISHED'}


class MESH2GS_OT_check_python(bpy.types.Operator):
    bl_idname = "mesh2gs.check_python"
    bl_translation_context = CTX
    bl_label = "Check"
    bl_description = "Run the set Python and check PyTorch, gsplat and the CUDA graphics card (up to a minute)"

    def execute(self, context):
        path = training.python_path(context)
        ok, message = training.check_python(path)
        training.CHECKED[path] = ok, message
        self.report({'INFO'} if ok else {'ERROR'}, tr(message))
        _redraw(context)
        return {'FINISHED'}


class MESH2GS_OT_toggle_source(bpy.types.Operator):
    bl_idname = "mesh2gs.toggle_source"
    bl_translation_context = CTX
    bl_label = "Source ↔ Gaussians"
    bl_description = "Switch visibility between the source objects and the result"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return context.mode == 'OBJECT' and preview.is_gaussian_object(context.active_object)

    def execute(self, context):
        obj = context.active_object
        show_source = not obj.hide_get()
        for src in source_objects(obj):
            src.hide_set(not show_source)
        obj.hide_set(show_source)
        return {'FINISHED'}


class MESH2GS_OT_export_ply(bpy.types.Operator, ExportHelper):
    bl_idname = "mesh2gs.export_ply"
    bl_translation_context = CTX
    bl_label = "Export PLY"
    bl_description = "Save the active object's Gaussians as a 3DGS PLY"
    bl_options = {'REGISTER'}

    filename_ext = ".ply"
    filter_glob: bpy.props.StringProperty(translation_context=CTX, default="*.ply", options={'HIDDEN'})
    axis_convention: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Axes",
        description="The file's coordinate system: where +X, +Y, +Z point",
        items=[(code, f"{code} — {name}", "") for code, name in axes.PRESETS.items()],
        default="RDF",
    )

    @classmethod
    def poll(cls, context):
        return preview.is_gaussian_object(context.active_object)

    def invoke(self, context, event):
        if not self.filepath:
            folder = os.path.dirname(bpy.data.filepath) or os.path.expanduser("~")
            self.filepath = os.path.join(folder, bpy.path.clean_name(context.active_object.name) + ".ply")
        return ExportHelper.invoke(self, context, event)

    def execute(self, context):
        try:
            gaussians = axes.convert(preview.read_object(context.active_object), self.axis_convention)
            ply.write(self.filepath, gaussians)
        except (KeyError, OSError) as err:
            self.report({'ERROR'}, tr(str(err)))
            return {'CANCELLED'}
        size = os.path.getsize(self.filepath) / 1e6
        self.report({'INFO'}, tr("Saved {} Gaussians, {:.1f} MB").format(spaced(len(gaussians)), size))
        return {'FINISHED'}


class MESH2GS_OT_export_spz(bpy.types.Operator, ExportHelper):
    bl_idname = "mesh2gs.export_spz"
    bl_translation_context = CTX
    bl_label = "Export SPZ"
    bl_description = ("Save the active object's Gaussians as compressed SPZ (Niantic); "
                      "the format fixes the axes (RUB), viewers convert them themselves")
    bl_options = {'REGISTER'}

    filename_ext = ".spz"
    filter_glob: bpy.props.StringProperty(translation_context=CTX, default="*.spz", options={'HIDDEN'})
    scale: bpy.props.FloatProperty(
        translation_context=CTX,
        name="Scale",
        description=("Multiply sizes when writing. SPZ stores coordinates in 0.24 mm steps within ±2048: "
                     "a model in centimeters or a few centimeters across is better scaled up"),
        default=1.0, min=1e-4, soft_max=1000.0,
    )

    @classmethod
    def poll(cls, context):
        return preview.is_gaussian_object(context.active_object)

    def invoke(self, context, event):
        if not self.filepath:
            folder = os.path.dirname(bpy.data.filepath) or os.path.expanduser("~")
            self.filepath = os.path.join(folder, bpy.path.clean_name(context.active_object.name) + ".spz")
        return ExportHelper.invoke(self, context, event)

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "scale")
        obj = context.active_object
        if preview.is_gaussian_object(obj):
            for line in spz_scale_notes(preview.read_object(obj).positions * self.scale):
                layout.label(text=line, icon='ERROR')

    def execute(self, context):
        try:
            g = preview.read_object(context.active_object)
            if self.scale != 1.0:
                g = GaussianSet(positions=g.positions * self.scale, scales=g.scales * self.scale,
                                rotations=g.rotations, opacities=g.opacities, colors=g.colors)
            gaussians = axes.convert(g, "RUB")
            spz.write(self.filepath, gaussians)
        except (KeyError, OSError, ValueError) as err:
            self.report({'ERROR'}, tr(str(err)))
            return {'CANCELLED'}
        for line in spz_scale_notes(gaussians.positions):
            self.report({'WARNING'}, line)
        size = os.path.getsize(self.filepath) / 1e6
        self.report({'INFO'}, tr("Saved {} Gaussians, {:.1f} MB").format(spaced(len(gaussians)), size))
        return {'FINISHED'}


def spz_scale_notes(positions):
    """Warnings about SPZ's fixed-point positions for a set of this size and place."""
    if len(positions) == 0:
        return []
    extent = float(np.max(positions.max(axis=0) - positions.min(axis=0)))
    reach = float(np.abs(positions).max())
    notes = []
    if reach >= spz.LIMIT:
        notes.append(tr("Coordinates reach {:.0f}: SPZ only holds ±{:.0f}, scale down").format(reach, spz.LIMIT))
    elif extent < spz.SMALL_EXTENT:
        notes.append(tr("The model is {:.1f} cm: SPZ's 0.24 mm step will smooth fine detail; "
                        "scale it up or save PLY").format(extent * 100))
    return notes


class MESH2GS_OT_import_ply(bpy.types.Operator, ImportHelper):
    bl_idname = "mesh2gs.import_ply"
    bl_translation_context = CTX
    bl_label = "Import PLY (3DGS)"
    bl_description = "Load splats from a 3DGS PLY (a KIRI result, say) with a viewport preview"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".ply"
    filter_glob: bpy.props.StringProperty(translation_context=CTX, default="*.ply", options={'HIDDEN'})
    axis_convention: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Axes",
        description="The file's coordinate system: where +X, +Y, +Z point",
        items=[(code, f"{code} — {name}", "") for code, name in axes.PRESETS.items()],
        default="RDF",
    )

    def execute(self, context):
        try:
            gaussians = axes.to_blender(ply.read(self.filepath), self.axis_convention)
        except (ValueError, OSError) as err:
            self.report({'ERROR'}, tr(str(err)))
            return {'CANCELLED'}
        place_import(context, self.filepath, gaussians, "PLY")
        self.report({'INFO'}, tr("Loaded {} Gaussians").format(spaced(len(gaussians))))
        return {'FINISHED'}


class MESH2GS_OT_import_spz(bpy.types.Operator, ImportHelper):
    bl_idname = "mesh2gs.import_spz"
    bl_translation_context = CTX
    bl_label = "Import SPZ"
    bl_description = "Load splats from compressed SPZ (versions 2 and 3) with a viewport preview"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = ".spz"
    filter_glob: bpy.props.StringProperty(translation_context=CTX, default="*.spz", options={'HIDDEN'})

    def execute(self, context):
        try:
            gaussians = axes.to_blender(spz.read(self.filepath), "RUB")  # SPZ fixes its axes
        except (ValueError, OSError, EOFError) as err:
            self.report({'ERROR'}, tr(str(err)))
            return {'CANCELLED'}
        place_import(context, self.filepath, gaussians, "SPZ")
        self.report({'INFO'}, tr("Loaded {} Gaussians").format(spaced(len(gaussians))))
        return {'FINISHED'}


def place_import(context, path, gaussians, tag):
    name = os.path.splitext(os.path.basename(path))[0]
    obj = preview.write_object(name, gaussians, tag, 0.0, context.scene.mesh2gs.preview_limit)
    if context.mode == 'OBJECT':
        for o in context.selected_objects:
            o.select_set(False)
        obj.select_set(True)
        context.view_layer.objects.active = obj
    return obj
