import bpy

from . import i18n, preview
from .i18n import CTX

METHODS = [
    ('M1', "M1 — analytic", "Points spread by surface area, color sampled at every point; no GPU"),
    ('M0', "M0 — KIRI scheme", "Fixed Gaussians per triangle, like Mesh 2 3DGS in KIRI 3DGS Render"),
    ('M2', "M2 — optimization", "3DGS training from scratch on rendered views; needs NVIDIA (CUDA) and a Python with gsplat"),
    ('M3', "M3 — hybrid", "M1 as the start + a short training on rendered views; needs NVIDIA (CUDA) and a Python with gsplat"),
]
AVAILABLE_METHODS = {'M0', 'M1', 'M2', 'M3'}
TRAINED_METHODS = {'M2', 'M3'}
BUDGET_METHODS = {'M1', 'M2', 'M3'}  # budget as a count or a density; M0 counts per triangle
BYTES_PER_GAUSSIAN = 68  # PLY, SH degree 0


def _update_preview(self, context):
    for obj in context.scene.objects:
        if preview.is_gaussian_object(obj):
            preview.set_preview_fraction(obj, self.preview_limit)


class Mesh2GSSettings(bpy.types.PropertyGroup):
    method: bpy.props.EnumProperty(translation_context=CTX, name="Method", items=METHODS, default='M1')
    budget_mode: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Budget",
        items=[('COUNT', "Count", "Set the total number of Gaussians"),
               ('DENSITY', "Density", "Set the number of Gaussians per square meter of surface")],
        default='COUNT',
    )
    budget: bpy.props.IntProperty(
        translation_context=CTX,
        name="Gaussians",
        description="How many Gaussians to build over all selected objects",
        default=200_000, min=1, soft_max=5_000_000,
    )
    density: bpy.props.FloatProperty(
        translation_context=CTX,
        name="Per m²",
        description="Gaussians per square meter of surface",
        default=20_000.0, min=1.0, soft_max=1_000_000.0,
    )
    per_triangle: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Per triangle",
        description="How many Gaussians to put in every triangle (KIRI actually puts 1)",
        items=[('1', "1", "Like KIRI"), ('3', "3", ""), ('4', "4", ""), ('6', "6", "")],
        default='1',
    )
    color_mode: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Color",
        items=[('ALBEDO', "Albedo", "Material color only, no light: the object can be relit"),
               ('LIT', "Scene light", "Bake the scene's lighting (Cycles) into the colors: the object looks "
                                      "as rendered; needed to fit it into a captured splat scene")],
        default='ALBEDO',
    )
    light_scene: bpy.props.PointerProperty(
        translation_context=CTX,
        name="Splat scene",
        description="A captured scene (imported PLY/SPZ) whose light to bake into the object",
        type=bpy.types.Object,
        poll=lambda self, obj: preview.is_gaussian_object(obj),
    )
    lit_samples: bpy.props.IntProperty(
        translation_context=CTX,
        name="Light samples",
        description="Cycles samples for the lighting bake: more is less noise and slower",
        default=64, min=1, soft_max=1024,
    )
    bake_procedural: bpy.props.BoolProperty(
        translation_context=CTX,
        name="Bake procedural",
        description="A procedural Base Color is baked into a texture by Cycles on a temporary copy of the object",
        default=True,
    )
    use_emission: bpy.props.BoolProperty(
        translation_context=CTX,
        name="Include glow",
        description="Add the material's Emission to the color: screens, indicators, lamps. Needs baking",
        default=True,
    )
    bake_resolution: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Resolution",
        description="Bake texture size per object",
        items=[('1024', "1024", ""), ('2048', "2048", ""), ('4096', "4096", "")],
        default='2048',
    )
    selected_faces_only: bpy.props.BoolProperty(
        translation_context=CTX,
        name="Selected faces only",
        description="In Edit Mode, convert only the selected faces",
        default=False,
    )
    overlap: bpy.props.FloatProperty(
        translation_context=CTX,
        name="Overlap k",
        description="Disc radius relative to the mean spacing between Gaussians",
        default=0.7, min=0.1, max=5.0,
    )
    sampling: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Sampling",
        items=[('stratified', "Stratified", "Even spacing over the surface, no clumps or holes"),
               ('random', "Random", "Independent random points; for comparison in experiments")],
        default='stratified',
    )
    edge_aware: bpy.props.BoolProperty(
        translation_context=CTX,
        name="Stay inside edges",
        description="Squash discs near sharp and boundary edges so they do not overhang",
        default=True,
    )
    sharp_angle: bpy.props.FloatProperty(
        translation_context=CTX,
        name="Sharp edge from",
        description="Angle between faces from which an edge counts as sharp",
        default=0.5235988, min=0.0, max=3.14159, subtype='ANGLE',
    )
    seed: bpy.props.IntProperty(
        translation_context=CTX,
        name="Seed",
        description="Random generator seed, for reproducible results",
        default=0, min=0,
    )
    preview_limit: bpy.props.IntProperty(
        translation_context=CTX,
        name="Show up to",
        description="How many Gaussians to draw in the viewport; export is not affected",
        default=300_000, min=1_000, soft_max=2_000_000, update=_update_preview,
    )
    m2_iterations: bpy.props.IntProperty(
        translation_context=CTX,
        name="Iterations",
        description="M2 training steps from a random start; 7000 as in the experiments",
        default=7000, min=10, soft_max=30_000,
    )
    m3_iterations: bpy.props.IntProperty(
        translation_context=CTX,
        name="Iterations",
        description="M3 refinement steps after M1; 1000 beat M1 on every benchmark model",
        default=1000, min=10, soft_max=10_000,
    )
    train_views: bpy.props.IntProperty(
        translation_context=CTX,
        name="Views",
        description="How many renders from cameras on a sphere around the objects the training sees",
        default=64, min=4, soft_max=256,
    )
    train_size: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Render size",
        description="Side of the square view render, in pixels",
        items=[('256', "256", ""), ('512', "512", ""), ('1024', "1024", "")],
        default='512',
    )
    hide_source: bpy.props.BoolProperty(
        translation_context=CTX,
        name="Hide source objects",
        description="Hide the source objects after converting, to see the result",
        default=True,
    )


def _update_language(self, context):
    i18n.apply(self.language)


class Mesh2GSPreferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    language: bpy.props.EnumProperty(
        translation_context=CTX,
        name="Add-on language",
        description=("Language of the Mesh2GS interface (Blender's own interface is not affected). "
                     "Blender 3.6 needs Interface → Translation → Interface turned on for it"),
        items=[('AUTO', "As Blender", "Russian if Blender's interface is Russian, English otherwise"),
               ('RU', "Русский", "Always Russian"),
               ('EN', "English", "Always English")],
        default='AUTO', update=_update_language,
    )
    python: bpy.props.StringProperty(
        translation_context=CTX,
        name="Python with gsplat",
        description=("Interpreter with PyTorch (CUDA) and gsplat installed; M2/M3 train in it. "
                     "Empty: the development environment in the repository, if there is one"),
        subtype='FILE_PATH',
    )

    env_dir: bpy.props.StringProperty(
        translation_context=CTX,
        name="Install to",
        description="Folder for the components of M2/M3 (≈5 GB; 9 GB free needed while installing). "
                    "Empty: the add-on's own user folder",
        subtype='DIR_PATH',
    )
    show_manual: bpy.props.BoolProperty(
        translation_context=CTX,
        name="Install by hand",
        description="Show the commands to make the training Python yourself",
        default=False,
    )

    def draw(self, context):
        from . import setup_env, training
        tr = i18n.tr
        layout = self.layout
        layout.row().prop(self, "language", expand=True)
        layout.separator()

        box = layout.box()
        box.label(text_ctxt=CTX, text="Components for M2, M3 and light from a splat scene", icon='PACKAGE')
        col = box.column(align=True)
        col.label(text_ctxt=CTX, text="PyTorch and gsplat train Gaussians on an NVIDIA graphics card.")
        col.label(text_ctxt=CTX, text="M0 and M1 work without them, on any computer.")
        job = setup_env.INSTALLER
        if job is not None and job.running:
            col = box.column(align=True)
            if hasattr(col, "progress"):  # Blender 4.0+
                col.progress(factor=job.progress, type='BAR', text=job.text)
            else:
                col.label(text=f"{job.text} ({job.progress:.0%})", translate=False)
            col.label(text=tr("{:.0f} s so far").format(job.seconds), translate=False)
            box.operator("mesh2gs.cancel_install", icon='CANCEL')
        elif not setup_env.supported():
            box.label(text_ctxt=CTX, text="Needs Windows or Linux with an NVIDIA graphics card", icon='ERROR')
        else:
            box.prop(self, "env_dir")
            folder = setup_env_folder(self)
            box.label(text=tr("Into {} · {:.1f} GB free").format(folder, setup_env.free_gb(folder)),
                      icon='DISK_DRIVE', translate=False)
            row = box.row()
            row.scale_y = 1.4
            row.operator("mesh2gs.install_env", icon='IMPORT')
            if job is not None and job.stage == "DONE":
                box.label(text=tr("Installed: {}").format(job.result), icon='CHECKMARK', translate=False)
            elif job is not None:
                box.label(text=f"{job.text}: {tr(job.error)}", icon='ERROR', translate=False)

        layout.separator()
        layout.prop(self, "python")
        path = training.python_path(context)
        row = layout.row()
        row.label(text=tr("In use: {}").format(path) if path else tr("No Python set: M2 and M3 are unavailable"),
                  icon='CHECKMARK' if path else 'ERROR', translate=False)
        row.operator("mesh2gs.check_python", icon='PLAY')
        if training.CHECKED.get(path):
            ok, message = training.CHECKED[path]
            layout.label(text_ctxt=CTX, text=message, icon='CHECKMARK' if ok else 'ERROR', translate=False)

        layout.prop(self, "show_manual", icon='TRIA_DOWN' if self.show_manual else 'TRIA_RIGHT', emboss=False)
        if self.show_manual:
            box = layout.box()
            col = box.column(align=True)
            for step in training.SETUP_STEPS:
                col.label(text_ctxt=CTX, text=step)
            row = box.row()
            row.operator("mesh2gs.copy_setup", icon='COPYDOWN')
            row.label(text=tr("Needs an NVIDIA graphics card, Python {} and ≈5 GB of disk")
                      .format(training.SETUP_PYTHON), translate=False)


def setup_env_folder(prefs):
    """Where the components go: the chosen folder, else the add-on's user folder (`prefs` is
    None when the add-on was registered by a script)."""
    import os
    if prefs is not None and prefs.env_dir:
        return bpy.path.abspath(prefs.env_dir)
    try:  # Blender 4.2+ extensions have a user folder of their own
        return bpy.utils.extension_path_user(__package__, path="env", create=False)
    except (AttributeError, ValueError):
        return os.path.join(bpy.utils.user_resource('SCRIPTS'), "mesh2gs_env")
